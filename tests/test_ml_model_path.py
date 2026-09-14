"""MODEL_PATH must be overridable via env var, the same way db.py's DB_PATH is.

Without this, model.pkl has nowhere to live on Fly: it's excluded from the
Docker build context (.dockerignore) and the app directory inside the
container isn't on the persistent volume, so a model trained locally never
reaches production and one trained via `fly ssh console` doesn't survive a
redeploy. fly.toml points MODEL_PATH at /data/model.pkl -- the same mounted
volume DB_PATH already uses -- so training once there persists it for good.
"""
import importlib
import os
import unittest

import ml


class TestModelPathIsConfigurable(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("MODEL_PATH", None)
        importlib.reload(ml)

    def test_env_var_overrides_the_default_path(self):
        os.environ["MODEL_PATH"] = "/data/model.pkl"
        importlib.reload(ml)
        self.assertEqual("/data/model.pkl", ml.MODEL_PATH)

    def test_default_path_is_local_to_the_app_directory_without_the_env_var(self):
        os.environ.pop("MODEL_PATH", None)
        importlib.reload(ml)
        self.assertTrue(ml.MODEL_PATH.endswith("model.pkl"))
        self.assertIn(os.path.dirname(os.path.abspath(ml.__file__)), ml.MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
