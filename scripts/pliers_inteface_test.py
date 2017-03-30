from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.pliers as pliers

from nipype.interfaces.utility import Function

import os

bids_dir = os.path.abspath('~/datasets/ds009')
task = 'emotionalrecognition'
run = 'run-01'
subject = '01'

"""
Set up workflow
"""
wf = Workflow(name='first_level')
wf.work_dir = '/tmp/pliers_test'

"""
Get event files
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
    output_names=['events'], function=_get_events))
events_getter.inputs.runs = run
events_getter.inputs.bids_dir = bids_dir
events_getter.inputs.task = task

"""
Extract features, for a given set of subjects, etc and write out a
set of new event files.
"""

pliers_extract = Node(interface=pliers.PliersInterface(), name='pliers')
pliers_extract.inputs.graph_spec = 'test_graph_path.json'


wf.connect(events_getter, 'events', pliers_extract, 'event_files')
wf.run()
