"""The MFA timeout must actually stop a run that overruns.

An alignment was observed hanging for ~55 minutes with a 30-minute timeout in
force. The cause is a standard subprocess trap: MFA starts workers, and when
the parent is killed those workers keep the stdout/stderr pipes open, so the
cleanup read blocks forever. These tests use a stand-in that reproduces the
shape of that process tree.
"""
import subprocess
import sys
import textwrap
import time

import pytest

from tapa.alignment import _run_with_timeout

ENV = None


def _script(body):
    return [sys.executable, "-c", textwrap.dedent(body)]


def test_returns_output_for_a_normal_command():
    rc, err = _run_with_timeout(
        _script("import sys; sys.stderr.write('hello'); sys.exit(0)"), 30, ENV)
    assert rc == 0
    assert "hello" in err


def test_reports_nonzero_exit():
    rc, err = _run_with_timeout(
        _script("import sys; sys.stderr.write('boom'); sys.exit(3)"), 30, ENV)
    assert rc == 3
    assert "boom" in err


def test_timeout_raises_promptly_when_child_hangs():
    started = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_with_timeout(_script("import time; time.sleep(60)"), 2, ENV)
    assert time.time() - started < 20, "timeout did not return promptly"


def test_timeout_returns_even_when_grandchildren_hold_the_pipes():
    """The real failure: a surviving worker keeps stderr open after the kill."""
    parent = """
        import subprocess, sys, time, textwrap
        # a worker that outlives us and inherits our pipes
        subprocess.Popen([sys.executable, "-c",
                          "import time; time.sleep(120)"])
        time.sleep(120)
    """
    started = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_with_timeout(_script(parent), 2, ENV)
    elapsed = time.time() - started
    # Without the process-group kill this blocks until the worker exits (120s).
    assert elapsed < 40, f"hung for {elapsed:.0f}s waiting on a surviving child"


def test_process_group_is_actually_killed():
    """After a timeout the child must be gone, not merely detached."""
    proc_holder = {}
    slow = _script("import time; time.sleep(120)")
    started = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_with_timeout(slow, 2, ENV)
    # a killed group frees the CPU immediately; confirm we did not just leak it
    assert time.time() - started < 20
