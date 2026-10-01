"""Long mounted workspaces must not exceed Unix-domain socket path limits."""
import json
from pathlib import Path
import sys

import pytest

from protein_design_mcp.dispatch.env import EnvDispatcher, EngineError
from protein_design_mcp.manifest.schema import EngineSpec


@pytest.mark.asyncio
async def test_long_workspace_supports_multiprocessing_ipc_and_preserves_temp_files(tmp_path, monkeypatch):
    workspace = tmp_path / ('long-mounted-workspace-' * 5)
    workspace.mkdir()
    monkeypatch.setenv('TMPDIR', str(workspace))
    monkeypatch.setenv('PROTEIN_MCP_KEEP_WORKDIR', '1')
    engine = EngineSpec(repo='ipc-probe', env='unused', entry=(sys.executable,))
    code = '''
import json, tempfile
from pathlib import Path
from multiprocessing.connection import Listener
with Listener() as listener:
    address = listener.address
    Path(tempfile.gettempdir(), 'retained.txt').write_text('preserve me')
    print(json.dumps({'address': address, 'tempdir': tempfile.gettempdir()}))
'''
    result = await EnvDispatcher(runner=None, scratch_root=workspace).run(engine, ['-c', code], timeout=10)
    report = json.loads(result.stdout)
    assert len(report['address'].encode()) < 108
    assert (result.workdir / '.engine-tmp' / 'retained.txt').read_text() == 'preserve me'
    assert not Path(report['tempdir']).is_symlink(), 'temporary alias must be removed after child exit'
    assert result.workdir.parent == workspace


@pytest.mark.asyncio
async def test_failed_engine_cleans_alias_but_retains_files(tmp_path, monkeypatch):
    workspace = tmp_path / ('long-mounted-workspace-' * 5)
    workspace.mkdir()
    monkeypatch.setenv('TMPDIR', str(workspace))
    monkeypatch.setenv('PROTEIN_MCP_KEEP_WORKDIR', '1')
    engine = EngineSpec(repo='ipc-probe', env='unused', entry=(sys.executable,))
    code = '''
import tempfile
from pathlib import Path
Path('alias.txt').write_text(tempfile.gettempdir())
Path(tempfile.gettempdir(), 'retained.txt').write_text('partial')
raise SystemExit(9)
'''
    with pytest.raises(EngineError) as error:
        await EnvDispatcher(runner=None, scratch_root=workspace).run(engine, ['-c', code], timeout=10)
    workdir = Path(error.value.execution_artifacts['workdir'])
    alias = Path((workdir / 'alias.txt').read_text())
    assert not alias.is_symlink()
    assert (workdir / '.engine-tmp/retained.txt').read_text() == 'partial'
