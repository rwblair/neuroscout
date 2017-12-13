import sys
sys.path.insert(0, "../")
from fmri_hcp_firstlevel import FirstLevelBIDS2


class AllFeatures(FirstLevelBIDS2):
    def _get_features(bids_dir, runs, subject_id, task_id):
        """ Inject extracted features into event files """
        import pandas as pd
        from glob import glob
        from os import path
        import numpy as np
        import re
        from functools import partial

        from bids.grabbids import BIDSLayout
        layout = BIDSLayout(bids_dir)
        event_files = [layout.get(
            type='events', return_type='file', subject=subject_id, run=r, task=task_id)[0] for r in runs]

        features = sorted(glob(
            '/mnt/c/Users/aid338/Documents/neuroscout_scripts/forrest_extract_results/clip*'))

        def setmaxConfidence(row):
            maxcol = row['maxFace']
            if pd.notnull(maxcol):
                val = row[maxcol]
            else:
                val = 0
            return val

        def polyArea(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        def computeMaxFaceArea(row):
            maxcol = row['maxFace']

            if pd.notnull(maxcol):
                prepend = re.sub('face_detectionConfidence', '', maxcol)
                x = []
                y = []

                for i in range(1, 5):
                    x.append(row[prepend + 'boundingPoly_vertex{}_x'.format(i)])
                    y.append(row[prepend + 'boundingPoly_vertex{}_y'.format(i)])

                val = polyArea(x, y)
            else:
                val = 0
            return val

        def calcNumFaces(confidence_cols, row):
            return (row[confidence_cols].isnull() == False).sum()

        new_event_files = []
        for i, run_file in enumerate(event_files):
            run_events = pd.read_table(run_file)
            run_features = pd.read_csv(features[i])

            # Move onset to when clip begins
            clip_start = run_events.iloc[0].onset
            run_features['onset'] = run_features['onset'] + clip_start

            # Calculate computed values
            confidence_cols = [c for c in run_features.columns if c.endswith(
                'face_detectionConfidence')]
            run_features['maxFace'] = run_features.apply(lambda x: x[confidence_cols].idxmax(), axis=1)
            run_features['maxfaceConfidence'] = run_features.apply(setmaxConfidence, axis=1)
            run_features['maxfaceArea'] = run_features.apply(computeMaxFaceArea, axis=1)
            partialnFaces = partial(calcNumFaces, confidence_cols)
            run_features['numFaces'] = run_features.apply(partialnFaces, axis=1)

            # Select only relevant columns
            run_features = run_features[['onset', 'duration', 'maxfaceConfidence', 'maxfaceArea', 'numFaces']]
            run_features = pd.melt(run_features, id_vars=['onset', 'duration'],
                                   value_name='amplitude', var_name='trial_type')

            # Save to curr dir, but give abs path
            base_file = path.splitext(path.basename(run_file))
            new_file = path.abspath('{}_fx{}'.format(*base_file))
            new_event_files.append(new_file)
            run_features.to_csv(new_file, sep=str('\t'), index=False)

    def validate_arguments(self, args):
        super(AllFeatures, self).validate_arguments(args)

        ### TODO: EDIT THIS!!!!
        self.field_template = dict(
            func='downsample/3/downsampled_func/%s/tfMRI_MOVIE%s*[AP]_flirt.nii.gz')
        self.template_args = dict(
            func=[['subject_id', 'runs']])


        conditions = ['street', 'outdoors', 'light', 'adult', 'sentiment',
                      'long_freq', 'concreteness', 'word',
                      '60_250',
                      'maxfaceConfidence']

        def create_contrasts(conditions):
            """ Creates basic contrasts.
            Expand this and stick into main script later """
            contrasts = []
            for i, con in enumerate(conditions):
                mat = [0] * len(conditions)
                mat[i] = 1
                c = [con, 'T', conditions, mat]
                contrasts.append(c)

            return contrasts
        self.arguments['conditions'] = conditions
        self.arguments['contrasts'] = create_contrasts(conditions)
        self.arguments['TR'] = 2
        self.arguments['task'] = 'movie'
