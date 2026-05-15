"""Smoke tests for package import."""


def test_import():
    import lwasolarutl as lsu

    assert lsu.__version__ == "0.1.0"
    assert hasattr(lsu, "file")
    assert hasattr(lsu, "ndfits")
    assert hasattr(lsu, "visualization")
    assert callable(lsu.ndfits.read)
    assert not hasattr(lsu, "processing")
    assert callable(lsu.recover_fits_from_h5)
    assert callable(lsu.compress_fits_to_h5)
    assert callable(lsu.visualization.slow_pipeline_default_plot)
