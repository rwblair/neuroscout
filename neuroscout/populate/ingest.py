""" Dataset ingestion
Tools to populate database from BIDS datasets
"""
import json
import magic
from flask import current_app
from pathlib import Path

import pandas as pd
import nibabel as nib
import numpy as np

from bids.grabbids import BIDSLayout
from bids.variables import load_variables
from datalad.api import install
from datalad.auto import AutomagicIO

from .utils import remote_resource_exists, format_preproc, hash_stim
from utils import listify, get_or_create
from models import (Dataset, Task, Run, Predictor, PredictorEvent, PredictorRun,
                    Stimulus, RunStimulus, GroupPredictor, GroupPredictorValue)
from database import db


def add_predictor(predictor_name, dataset_id, run_id, onsets, durations, values,
                  source=None, **kwargs):
    """" Adds a new Predictor to a run given a set of values
    If Predictor already exist, use that one
    Args:
        predictor_name - name given to predictor_name
        dataset_id - dataset db id
        run_id - run db id
        onsets - list of onsets
        durations - list of durations
        values - list of values
        source - source of event
        kwargs - additional identifiers of the Predictor
    Output:
        predictor id

    """
    predictor, _ = get_or_create(
        Predictor, name=predictor_name, dataset_id=dataset_id,
        source=source, **kwargs)

    values = pd.Series(values, dtype='object')
    # Insert each row of Predictor as PredictorEvent
    for i, val in enumerate(values[values!='n/a']):
        pe, _ = get_or_create(
            PredictorEvent, commit=False,
            onset=onsets[i], predictor_id=predictor.id, run_id = run_id)
        pe.duration = durations[i]
        pe.value = str(val)
        db.session.commit()

    # Add PredictorRun
    pr, _ = get_or_create(
        PredictorRun, predictor_id=predictor.id, run_id = run_id)

    return predictor.id


def add_predictor_collection(collection, ds_id, run_id, TR=None, include=None):
    """ Add a RunNode to the database.
    Args:
        collection - BIDSVariableCollection to ingest
        ds_id - Dataset model id
        r_id - Run model id
        source - source of collection. e.g "events", "recordings"
        TR - time repetiton of task
        include - list of predictors to include. all if None.
    """
    for name, var in collection.variables.items():
        if include is not None and name not in include:
            break
        values = var.values.values.tolist()
        if hasattr(var, 'onset'):
            onset = var.onset.tolist()
            duration = var.duration.tolist()
        else:
            if TR is not None:
                var = var.resample(1 / TR)
            else:
                TR = var.sampling_rate / 2

            onset = np.arange(0, len(var.values) * TR, TR).tolist()
            duration = [(TR)] * len(var.values)

        add_predictor(name, ds_id, run_id, onset, duration, values, var.source)


def add_group_predictors(dataset_id, participants):
    """ Adds group predictors using participants.tsv
    Args:
        participants - path to participants tsv
        dataset_id - Dataset model id
        subjects - subject ids to processed
    Output:
        Ids of group predictors added
    """
    gp_ids = []
    try:
        from os.path import isfile
        assert isfile(participants.as_posix())
        participants = pd.read_csv(participants, delimiter='\t')
    except:
        return []

    participants = dict(participants.iteritems())
    subs = participants.pop('participant_id')

    # Parse participant columns and insert as GroupPredictors
    for col in participants.keys():
        gp, _ = get_or_create(
            GroupPredictor, name=col, dataset_id=dataset_id, level='subject')

        for i, val in participants[col].items():
            sub_id = subs[i].split('-')[1]
            subject_runs = Run.query.filter_by(dataset_id=dataset_id,
                                               subject=sub_id)
            for run in subject_runs:
                gpv, _ = get_or_create(
                    GroupPredictorValue, commit=False,
                    gp_id=gp.id, run_id = run.id, level_id=sub_id)
                gpv.value = str(val)
        db.session.commit()
        gp_ids.append(gp.id)

    return gp_ids


def add_stimulus(path, stim_hash, dataset_id, parent_id=None,
                 converter_name=None, converter_params=None):
    """ Creare stimulus model """
    path = path.resolve().as_posix()
    mimetype = magic.from_file(path, mime=True)

    model, new = get_or_create(
        Stimulus, commit=False, sha1_hash=stim_hash,
        dataset_id=dataset_id, converter_name=converter_name)

    if new:
        model.path = path
        model.mimetype = mimetype
        model.parent_id = parent_id
        model.converter_params = converter_params
        db.session.commit()

    return model, new


def add_task(task_name, dataset_name=None, local_path=None,
             dataset_address=None, preproc_address=None, include_predictors=None,
             reingest=False, scan_length=1000, automagic=False, **kwargs):
    """ Adds a BIDS dataset task to the database.
        Args:
            task_name - task to add
            dataset_name - overide dataset name
            local_path - path to local bids dataset.
            dataset_address - remote address of BIDS dataset.
            preproc_address - remote address of preprocessed files.
            include_predictors - set of predictors to ingest
            reingest - force reingesting even if dataset already exists
            scan_length - default scan length in case it cant be found in image
            automagic - force enable DataLad automagic
            kwargs - arguments to filter runs by
        Output:
            dataset model id
    """

    if dataset_address is not None and local_path is None:
        local_path = install(
            source=dataset_address,
            path=(Path(
                current_app.config['DATASET_DIR']) / dataset_name).as_posix()
            ).path
        automagic = True

    if automagic:
        automagic = AutomagicIO()
        automagic.activate()

    from os.path import isfile

    local_path = Path(local_path)

    # Look for events folder in derivatives
    path = [local_path.as_posix()]
    extra_events = local_path / 'derivatives' / 'events'
    if extra_events.exists():
        path.append(extra_events.as_posix())

    # Remove once grabbit bug fix is in
    if len(path) == 1:
        path = path[0]

    layout = BIDSLayout(path)
    if task_name not in layout.get_tasks():
        raise ValueError("Task {} not found in dataset {}".format(
            task_name, local_path))

    description = json.load((local_path / 'dataset_description.json').open())
    dataset_name = dataset_name if dataset_name is not None \
                   else description['Name']

    # Get or create dataset model from mandatory arguments
    dataset_model, new_ds = get_or_create(Dataset, name=dataset_name)

    if new_ds:
        dataset_model.description = description
        dataset_model.dataset_address = dataset_address
        dataset_model.preproc_address = preproc_address
        dataset_model.local_path = local_path.as_posix()
        db.session.commit()
    elif not reingest:
        if automagic:
            automagic.deactivate()
        return dataset_model.id

    # Get or create task
    task_model, new_task = get_or_create(
        Task, name=task_name, dataset_id=dataset_model.id)

    if new_task:
        task_model.description = json.load(
            (local_path / 'task-{}_bold.json'.format(task_name)).open())
        task_model.TR = task_model.description['RepetitionTime']
        db.session.commit()

    stims_processed = {}
    """ Parse every Run """
    for img in layout.get(task=task_name, type='bold', extensions='.nii.gz',
                          **kwargs):
        current_app.logger.info("Processing task {} subject {}, run {}".format(
            img.task, img.subject, img.run))

        # Get entities
        entities = {entity : getattr(img, entity)
                    for entity in ['subject', 'session']
                    if entity in img._fields}

        """ Extract Run information """
        run_model, new = get_or_create(
            Run, dataset_id=dataset_model.id, number=img.run,
            task_id = task_model.id, **entities)
        entities['run'] = img.run
        entities['task'] = task_model.name

        # Get duration (helps w/ transformations)
        try:
            img_ni = nib.load(img.filename)
            run_model.duration = img_ni.shape[3] * img_ni.header.get_zooms()[-1] \
                                    / 1000
        except (nib.filebasedimages.ImageFileError, IndexError) as e:
            current_app.logger.debug(
                "Error loading BOLD file, default duration used.")
            run_model.duration = scan_length

        run_model.func_path = format_preproc(suffix="preproc", **entities)
        run_model.mask_path = format_preproc(suffix="brainmask", **entities)

        # Confirm remote address exists:
        if preproc_address is not None:
            remote_resource_exists(preproc_address, run_model.func_path)
            remote_resource_exists(preproc_address, run_model.mask_path)

        """ Extract Predictors"""
        current_app.logger.info("Extracting predictors")

        # Assert event files exist (for DataLad)
        for e in listify(layout.get_events(img.filename)):
            assert isfile(e)

        collection = load_variables(
            layout, levels='run', scan_length=run_model.duration,  **entities).\
            get_collections('run')[0]

        stims = collection.variables.pop('stim_file')

        add_predictor_collection(
            collection, dataset_model.id, run_model.id,
            include=include_predictors, TR=task_model.TR)

        """ Ingest Stimuli """
        current_app.logger.info("Ingesting stimuli")

        for i, val in enumerate(stims.values):
            stim_path = local_path / 'stimuli' / val
            if val not in stims_processed:
                try:
                    stim_hash = hash_stim(stim_path)
                except OSError:
                    current_app.logger.debug('{} not found.'.format(val))
                    continue

                stims_processed[val] = stim_hash
            else:
                stim_hash = stims_processed[val]

            stim_model, _ = add_stimulus(
                stim_path, stim_hash, dataset_id=dataset_model.id)

            # Get or create Run Stimulus association
            runstim, _ = get_or_create(
                RunStimulus, stimulus_id=stim_model.id, run_id=run_model.id,
                onset=stims.onset.tolist()[i],
                duration=stims.duration.tolist()[i])

    """ Add GroupPredictors """
    current_app.logger.info("Adding group predictors")
    add_group_predictors(dataset_model.id, local_path / 'participants.tsv')

    if automagic:
        automagic.deactivate()

    return dataset_model.id