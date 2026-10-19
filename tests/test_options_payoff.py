from pathlib import Path
import shutil
import subprocess
import pytest


def test_options_payoff_node_suites():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for payoff browser-model checks')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, '--test', 'tests/options_payoff.test.cjs',
         'tests/options_payoff_simulator.test.cjs'],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
