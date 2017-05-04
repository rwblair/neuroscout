from tests.request_utils import decode_json
def test_get_predictor(auth_client, add_dataset):
    # List of predictors
    rv = auth_client.get('/api/predictors')
    assert rv.status_code == 200
    pred_list = decode_json(rv)
    assert type(pred_list) == list

    # Get first predictors
    first_pred_id = pred_list[0]['id']
    assert 'name' in pred_list[0]

    # Get first predictors by id
    rv = auth_client.get('/api/predictors/{}'.format(first_pred_id))
    assert rv.status_code == 200
    pred = decode_json(rv)
    assert first_pred_id == pred['id']

    # Try getting nonexistent predictor
    rv = auth_client.get('/api/predictors/{}'.format('123'))
    assert rv.status_code == 404

    # Test parameters
    ds = decode_json(
        auth_client.get('/api/datasets'))
    run_id = str(ds[0]['runs'][0])
    rv = auth_client.get('/api/predictors', params={'run_id' : run_id})
    assert rv.status_code == 200
    pred_select = decode_json(rv)
    assert type(pred_select) == list


    rv = auth_client.get('/api/predictors', params={'run_id' : '123123'})
    assert rv.status_code == 200
    assert len(decode_json(rv)) == 0


    # Test filtering by multiple parameters
    rv = auth_client.get('/api/predictors', params={'name': 'rt',
        'run_id': run_id})
    assert rv.status_code == 200
    pred_p = decode_json(rv)
    assert len(pred_p) == 1

    # Get PredictorEvent List
    rv = auth_client.get('/api/predictor-events')
    assert rv.status_code == 200
    pe_list = decode_json(rv)
    assert type(pe_list) == list

    # Get PredictorEvent w/ params
    rv = auth_client.get('/api/predictor-events',
    params={'predictor_name' : 'rt',
        'run_id': run_id})
    assert rv.status_code == 200
    pe_list = decode_json(rv)
    assert type(pe_list) == list
    assert len(pe_list) == 4
    pe_id = pe_list[0]['id']

    # Get PredictorEvent by id
    rv = auth_client.get('/api/predictor-events/{}'.format(pe_id))
    assert rv.status_code == 200
