"""Cross-domain job registry: same recipe per model, only the source data differs."""
from src.domains import DOMAINS
from src.experiments import CROSS_DOMAIN_BUDGETS, CROSS_DOMAIN_MODELS, JOBS, cross_domain_job_id


def test_matrix_covers_every_source_and_model():
    ids = {cross_domain_job_id(s, m) for s in DOMAINS for m in CROSS_DOMAIN_MODELS}
    assert len(ids) == 15 and ids <= set(JOBS)


def test_kaggle_reuses_existing_jobs():
    assert cross_domain_job_id('kaggle', 'unet') == 'unet_edge_aug'
    assert cross_domain_job_id('kaggle', 'kd') == 'kd_cnn'
    assert cross_domain_job_id('kaggle', 'cnn') == 'xd_kaggle_cnn'


def test_recipe_matches_kaggle_except_data():
    skip = {'job_id', 'data', 'kd_teacher'}
    for m in CROSS_DOMAIN_MODELS:
        ref = JOBS[cross_domain_job_id('kaggle', m)].to_dict()
        for s in ('augraphy', 'dibco'):
            job = JOBS[cross_domain_job_id(s, m)]
            assert job.data == s
            assert {k: v for k, v in job.to_dict().items() if k not in skip} == \
                   {k: v for k, v in ref.items() if k not in skip}


def test_kd_teacher_same_source_and_budgeted_first():
    order = list(CROSS_DOMAIN_BUDGETS)
    for s in ('augraphy', 'dibco'):
        kd = JOBS[cross_domain_job_id(s, 'kd')]
        assert kd.kd_teacher == cross_domain_job_id(s, 'unet')
        assert order.index(kd.kd_teacher) < order.index(kd.job_id)


def test_cnn_baseline_has_no_teacher():
    for s in DOMAINS:
        assert JOBS[cross_domain_job_id(s, 'cnn')].kd_teacher is None


def test_budgets_only_new_jobs():
    assert set(CROSS_DOMAIN_BUDGETS) == {j for j in JOBS if j.startswith('xd_')}
    assert len(CROSS_DOMAIN_BUDGETS) == 11
