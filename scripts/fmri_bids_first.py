""""
Usage:
    fmri_bids_first run [options] <bids_dir> <task> <model> <out_dir>
    fmri_bids_first make [options] <bids_dir> <task> <model> [<out_dir>]

-t <transformations>    Transformation JSON spec to apply.
-p <pliers_graph>       Pliers JSON spec of features to extract.
-s <subject_id>         Subjects to analyze. [default: all]
-r <run_ids>            Runs to analyze. [default: all]
-w <work_dir>           Working directory.
-c                      Stop on first crash.
--jobs=<n>              Number of parallel jobs [default: 1].
"""

import os
import json
from docopt import docopt
from nipype.pipeline.engine import Node, Workflow
from nipype.workflows.fmri.fsl import (create_modelfit_workflow,
                                       create_fixed_effects_flow)

import nipype.algorithms.modelgen as modelgen
import nipype.algorithms.events as events

from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import Function, IdentityInterface
import nipype.interfaces.pliers as pliers

from bids.grabbids import BIDSLayout

def validate_arguments(args):
    """
    Clean up argument names
    """
    var_names = {'<out_dir>': 'out_dir',
                 '<task>': 'task',
                 '<in_dir>': 'in_dir',
                 '<bids_dir>': 'bids_dir',
                 '<model>': 'model',
                 '-w': 'work_dir',
                 '-s': 'subjects',
                 '-r': 'runs',
                 '-t': 'transformations',
                 '-p' : 'pliers_graph'}

    for old, new in var_names.items():
        if old in args:
            args[new] = args.pop(old)

    """
    BIDS specific validation.
    """
    layout = BIDSLayout(args['bids_dir'])

    # Check BOLD data
    if 'bold' not in layout.get_types():
        raise Exception("BIDS project does not contain"
                        " preprocessed BOLD data.")

    # Check that task exists
    if args['task'] not in layout.get_tasks():
        raise Exception("Task not found in BIDS project")

    # Check subject ids and runs
    for entity in ['subjects', 'runs']:
        # Parse
        args[entity] = args[entity].split(" ")

        all_ents = layout.get(
            target=entity[:-1], return_type='id', task=args['task'])

        if args[entity] == 'all':
            args[entity] = all_ents
        else:
            for e in args[entity]:
                if e not in all_ents:
                    raise Exception("Invalid {} id {}.".format(entity[:-1], e))



    if args.pop('-c'):
        from nipype import config
        cfg = dict(logging=dict(workflow_level='DEBUG'),
                   execution={'stop_on_first_crash': True})
        config.update_config(cfg)


    for directory in ['out_dir', 'work_dir']:
        if args[directory] is not None:
            args[directory] = os.path.abspath(args[directory])
            if not os.path.exists(args[directory]):
                os.makedirs(args[directory])

    args['in_dir'] = os.path.join(
        os.path.abspath(args['bids_dir']), 'derivatives/fmriprep')

    # JSON validation
    for json_name in ['pliers_graph', 'transformations']:
        if args[json_name] is not None:
            args[json_name] = os.path.abspath(args[json_name])
            try:
                json.load(open(args[json_name], 'r'))
            except ValueError:
                raise Exception("Invalid {}} JSON file".format(json_name))
            except IOError:
                raise Exception("JSON file not found: {}".format(json_name))

    return args

def create_first_level(bids_dir, in_dir, task, subjects, runs, model,
                       out_dir=None, work_dir=None, transformations=None,
                       pliers_graph=None, TR=1):
    """
    Set up workflow
    """
    wf = Workflow(name='first_level')
    if work_dir:
        wf.base_dir = work_dir

    """
    Subject iterator
    """
    infosource = Node(IdentityInterface(fields=['subject_id']),
                      name="infosource")
    infosource.iterables = ('subject_id', subjects)

    """
    Grab data for each subject
    """

    datasource = Node(DataGrabber(infields=['subject_id'],
                                  outfields=['func']),
                      name='datasource')
    datasource.inputs.base_directory = in_dir
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    datasource.inputs.field_template =  dict(
        func='sub-%s/func/sub-%s_task-%s_%s_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
    datasource.inputs.template_args = dict(
        func=[['subject_id', 'subject_id', 'task', 'runs']])
    datasource.inputs.runs = arguments['runs']
    datasource.inputs.task = arguments['task']

    """
    Add BIDS events and connect to workflow.
    """
    def _get_events(bids_dir, subject_id, runs, task):
        """ Get a subject's event files """
        from bids.grabbids import BIDSLayout
        layout = BIDSLayout(bids_dir)
        events = [layout.get(
            type='events', return_type='file', subject=subject_id, run=r, task=task)[0] for r in runs]
        return events

    events_getter = Node(name='events', interface=Function(
        input_names=['bids_dir', 'subject_id', 'runs', 'task'],
        output_names=['event_files'], function=_get_events))
    events_getter.inputs.runs = arguments['runs']
    events_getter.inputs.bids_dir = arguments['bids_dir']
    events_getter.inputs.task = arguments['task']

    """
    Extract features using pliers and add to events.tsv.
    """

    pliers_extract = Node(interface=pliers.PliersInterface(), name='pliers')
    if pliers_graph is not None:
        pliers_extract.inputs.graph_sec = pliers_graph

    """
    Specify model, apply transformations and specify fMRI model
    """

    eventspec = Node(interface=events.SpecifyEvents(), name="eventspec")
    modelspec = Node(interface=modelgen.SpecifyModel(), name="modelspec")

    eventspec.inputs.input_units = 'secs'
    eventspec.inputs.time_repetition = TR
    if transformations is not None:
        eventspec.inputs.transformations = transformations

    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = TR
    modelspec.inputs.high_pass_filter_cutoff = 100.

    wf.connect([(infosource, datasource, [('subject_id', 'subject_id')]),
                (infosource, events_getter, [('subject_id', 'subject_id')]),
                (events_getter, pliers_extract, [('event_files', 'event_files')]),
                (pliers_extract, eventspec, [('event_files', 'event_files')]),
                (datasource, modelspec, [('func', 'functional_runs')]),
                (eventspec, modelspec, [('subject_info', 'subject_info')])])

    """
    Fit model to each run
    """

    modelfit = create_modelfit_workflow()
    contrasts = None #### TEMPORARY
    modelfit.inputs.inputspec.contrasts = contrasts
    modelfit.inputs.inputspec.interscan_interval = TR
    modelfit.inputs.inputspec.model_serial_correlations = True
    modelfit.inputs.inputspec.bases = {'gamma': {'derivs': True}}

    wf.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
    wf.connect(datasource, 'func', modelfit, 'inputspec.functional_data')

    """
    Fixed effects workflow to combine runs
    """

    fixed_fx = create_fixed_effects_flow()

    def sort_copes(copes, varcopes, contrasts):
        import numpy as np
        if not isinstance(copes, list):
            copes = [copes]
            varcopes = [varcopes]
        num_copes = len(contrasts)
        n_runs = len(copes)
        all_copes = np.array(copes).flatten()
        all_varcopes = np.array(varcopes).flatten()
        outcopes = all_copes.reshape(int(len(all_copes) / num_copes),
                                     num_copes).T.tolist()
        outvarcopes = all_varcopes.reshape(int(len(all_varcopes) / num_copes),
                                           num_copes).T.tolist()
        return outcopes, outvarcopes, n_runs

    cope_sorter = Node(Function(input_names=['copes', 'varcopes',
                                             'contrasts'],
                                output_names=['copes', 'varcopes',
                                              'n_runs'],
                                function=sort_copes),
                       name='cope_sorter')
    cope_sorter.inputs.contrasts = contrasts

    wf.connect([(modelfit, cope_sorter, [('outputspec.copes', 'copes')]),
                (modelfit, cope_sorter, [('outputspec.varcopes', 'varcopes')]),
                (cope_sorter, fixed_fx, [('copes', 'inputspec.copes'),
                                         ('varcopes', 'inputspec.varcopes'),
                                         ('n_runs', 'l2model.num_copes')]),
                (modelfit, fixed_fx, [('outputspec.dof_file',
                                       'inputspec.dof_files'),
                                      ])
                ])
    """
    Save to datasink
    """

    def get_subs(subject_id, conds):
        """ Generate substitutions """
        subs = [('_subject_id_%s' % subject_id, '')]

        for i in range(len(conds)):
            subs.append(('_flameo%d/cope1.' % i, 'cope%02d.' % (i + 1)))
            subs.append(('_flameo%d/varcope1.' % i, 'varcope%02d.' % (i + 1)))
            subs.append(('_flameo%d/zstat1.' % i, 'zstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/tstat1.' % i, 'tstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/res4d.' % i, 'res4d%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_warp.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_warp.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_warp.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_trans.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_trans.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_trans.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
        return subs

    subsgen = Node(Function(input_names=['subject_id', 'conds'],
                            output_names=['substitutions'],
                            function=get_subs),
                   name='subsgen')

    datasink = Node(DataSink(), name="datasink")
    if out_dir is not None:
        datasink.inputs.base_directory = os.path.abspath(out_dir)

    wf.connect(infosource, 'subject_id', datasink, 'container')
    wf.connect(infosource, 'subject_id', subsgen, 'subject_id')
    wf.connect(subsgen, 'substitutions', datasink, 'substitutions')
    subsgen.inputs.conds = contrasts

    wf.connect([(modelfit.get_node('modelgen'), datasink,
                 [('design_cov', 'qa.model'),
                  ('design_image', 'qa.model.@matrix_image'),
                  ('design_file', 'qa.model.@matrix'),
                  ])])

    wf.connect([(fixed_fx.get_node('outputspec'), datasink,
                 [('res4d', 'res4d'),
                  ('copes', 'copes'),
                  ('varcopes', 'varcopes'),
                  ('zstats', 'zstats'),
                  ('tstats', 'tstats')])
                ])


    return wf


if __name__ == '__main__':
    arguments = validate_arguments(docopt(__doc__))
    jobs = arguments.pop('--jobs')
    run = arguments.pop('run')
    arguments.pop('make')
    wf = create_first_level(**arguments)

    if run:
        if jobs == 1:
            wf.run()
        else:
            wf.run(plugin='MultiProc', plugin_args={'n_procs': jobs})
