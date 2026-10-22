import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_all_python_parses():
    for path in (ROOT/'src').rglob('*.py'):
        ast.parse(path.read_text(),filename=str(path))


@pytest.mark.parametrize('entry',['training','testing','seg_training','seg_testing','run_kfold'])
def test_help(entry,tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'src'/f'{entry}.py'),'--help'],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout
    assert not list(tmp_path.iterdir())


def test_import_has_no_task_side_effects(tmp_path):
    code="import sys; sys.path.insert(0,sys.argv[1]); import training,testing,seg_training,seg_testing,run_kfold"
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'src')],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not list(tmp_path.iterdir())


def test_no_legacy_dispatch():
    for path in (ROOT/'src').rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('utils.header','utils.helper_funcs')
                assert all(n.name!='*' for n in node.names)


def test_kfold_commands(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'src'))
    import run_kfold
    import yaml
    commands=[]
    monkeypatch.setattr(run_kfold,'run_command',lambda cmd:commands.append(cmd))
    for task,config in [('diffusion','mlffm.yaml'),('seg','unet.yaml')]:
        commands.clear()
        monkeypatch.setattr(sys,'argv',['run_kfold','-c',str(ROOT/'configs/paper/isic2018'/config),
                                      '--folds','0,2','--config_dir',str(tmp_path),
                                      '--metrics_dir',str(tmp_path)])
        run_kfold.main()
        assert len(commands)==4
        assert commands[0][1]==('src/training.py' if task=='diffusion' else 'src/seg_training.py')
        assert commands[1][1]==('src/testing.py' if task=='diffusion' else 'src/seg_testing.py')
        cfg=yaml.safe_load(Path(commands[0][-1]).read_text())
        assert cfg['run']['kfold']['fold']==0
        assert cfg['model']['name'].endswith('_f0')


def test_kfold_output_dirs_from_config(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'src'))
    import run_kfold
    import yaml
    commands=[]
    monkeypatch.setattr(run_kfold,'run_command',lambda cmd:commands.append(cmd))
    monkeypatch.setattr(run_kfold,'read_metrics',lambda metrics_dir,name:{'Dice':1.0})
    cfg=yaml.safe_load((ROOT/'configs/paper/isic2018/unet.yaml').read_text())
    cfg['run']['kfold']={'metrics_dir':str(tmp_path/'metrics'),'config_dir':str(tmp_path/'configs')}
    path=tmp_path/'base.yaml';path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys,'argv',['run_kfold','-c',str(path),'--folds','0'])
    run_kfold.main()
    fold_cfg_path=Path(commands[0][-1])
    assert fold_cfg_path.parent==tmp_path/'configs'
    assert yaml.safe_load(fold_cfg_path.read_text())['run']['kfold']['metrics_dir']==str(tmp_path/'metrics')
    assert (tmp_path/'metrics'/'unet_i01_kfold_summary.json').exists()


def test_kfold_folds_from_config(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'src'))
    import run_kfold
    import yaml
    commands=[]
    monkeypatch.setattr(run_kfold,'run_command',lambda cmd:commands.append(cmd))
    cfg=yaml.safe_load((ROOT/'configs/paper/isic2018/unet.yaml').read_text())
    cfg['run']['kfold']={'folds':[2],'metrics_dir':str(tmp_path),'config_dir':str(tmp_path)}
    path=tmp_path/'base.yaml';path.write_text(yaml.safe_dump(cfg))
    for argv,expected in [([],[2]),(['--folds','0,1'],[0,1])]:
        commands.clear()
        monkeypatch.setattr(sys,'argv',['run_kfold','-c',str(path),'--skip_test',*argv])
        run_kfold.main()
        assert [yaml.safe_load(Path(c[-1]).read_text())['run']['kfold']['fold'] for c in commands]==expected


def test_kfold_aggregate_only(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'src'))
    import json
    import run_kfold
    import pytest
    commands=[]
    monkeypatch.setattr(run_kfold,'run_command',lambda cmd:commands.append(cmd))
    for fold in (0,1):
        (tmp_path/f'unet_i01_f{fold}_metrics.json').write_text(json.dumps({'metrics/Dice':0.8+fold/10}))
    argv=['run_kfold','-c',str(ROOT/'configs/paper/isic2018/unet.yaml'),'--metrics_dir',str(tmp_path),'--aggregate_only']
    monkeypatch.setattr(sys,'argv',argv+['--folds','0,1'])
    run_kfold.main()
    assert commands==[]
    summary=json.loads((tmp_path/'unet_i01_kfold_summary.json').read_text())
    assert abs(summary['metrics/Dice']['mean']-0.85)<1e-9
    monkeypatch.setattr(sys,'argv',argv+['--folds','0,1,2'])
    with pytest.raises(FileNotFoundError):
        run_kfold.main()


@pytest.mark.parametrize('model',['MLFFMSegDiff','DermoSegDiff','UNet','GenericUNet','SwinUNETR'])
def test_entrypoint_roundtrip(model,tmp_path):
    """Train two epochs without held-out evaluation, then test the last weights."""
    import numpy as np
    import yaml
    diffusion=model in ('MLFFMSegDiff','DermoSegDiff')
    files={'MLFFMSegDiff':'mlffm.yaml','DermoSegDiff':'dermosegdiff.yaml','UNet':'unet.yaml',
           'GenericUNet':'generic_unet.yaml','SwinUNETR':'swinunetr.yaml'}
    cfg=yaml.safe_load((ROOT/'configs/paper/isic2018'/files[model]).read_text())
    cfg['run']['device']='cpu'
    cfg['run']['continue_training']=False
    cfg['run']['writer_dir']=str(tmp_path/'events')
    cfg['run']['log_dir']=str(tmp_path/'run_logs')
    cfg['run']['kfold']={'enabled':True,'n_splits':5,'fold':0,'seed':0,
                        'metrics_dir':str(tmp_path/'metrics')}
    cfg['model']['name']=model
    cfg['model']['save_dir']=str(tmp_path/'models')
    (tmp_path/'models').mkdir()
    cfg['dataset']['data_dir']=str(tmp_path/'data')
    cfg['dataset']['input_size']=128
    cfg['augmentation']['p']=0
    cfg['training']['epochs']=2
    cfg['training']['save_every']=50  # the last epoch is saved regardless
    if diffusion:
        cfg['model']['params'].update(dim_x=16,dim_g=16,dim_x_mults=[1]*6,dim_g_mults=[1]*6)
        cfg['diffusion']['schedule']['timesteps']=10
        cfg['training']['intial_weights']['use']=False
        cfg['training']['loss']={'l2':{'cofficient':1}}
        cfg['testing']['ensemble']=1
        cfg['testing']['model_weigths']['overload']=False
        cfg['testing']['result_imgs']['save']=False
    elif model=='UNet':
        cfg['model']['params']['features']=[8,16]
    elif model=='GenericUNet':
        cfg['model']['params'].update(base_num_features=8,num_pool=2)
    else:
        cfg['model']['params'].update(feature_size=12,depths=[1]*4)
    for section in cfg['data_loader'].values():
        section.update(batch_size=2,num_workers=0,pin_memory=False)
    cache=tmp_path/'data/np';cache.mkdir(parents=True)
    rng=np.random.default_rng(4)
    np.save(cache/'X_tr_128x128.npy',rng.random((10,3,128,128),dtype=np.float32))
    np.save(cache/'Y_tr_128x128.npy',(rng.random((10,1,128,128))>.5).astype(np.float32))
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for entry in (['training','testing'] if diffusion else ['seg_training','seg_testing']):
        command=[sys.executable,str(ROOT/'src'/f'{entry}.py'),'-c',str(path)]
        if entry in ('training','seg_training'):
            guard = '''
import importlib, sys
sys.path.insert(0, sys.argv.pop(1))
m = importlib.import_module(sys.argv.pop(1))
original = m.get_dataloaders
def training_only(config, modes, **kwargs):
    assert modes == 'tr', 'Training requested held-out data'
    return original(config, modes, **kwargs)
m.get_dataloaders = training_only
def forbidden(*args, **kwargs):
    raise AssertionError('Held-out validation called during training')
m.validate_one_epoch = forbidden
m.validate = forbidden
import train_validate
train_validate.validate = forbidden
m.main()
'''
            command=[sys.executable,'-c',guard,str(ROOT/'src'),entry,'-c',str(path)]
        result=subprocess.run(command,
                              cwd=tmp_path,capture_output=True,text=True,env=env,timeout=180)
        assert result.returncode==0,result.stdout[-2000:]+result.stderr[-4000:]
    import json
    values=json.loads((tmp_path/'metrics'/f'{model}_metrics.json').read_text())
    assert len(values)==7
    assert all(np.isfinite(v) for v in values.values())
    assert len(list((tmp_path/'models').glob('*.pth')))==1
    assert len(list((tmp_path/'run_logs').glob(f'{model}*.log')))==2
    assert not (tmp_path/'logs').exists()
    import torch
    checkpoint=torch.load(next((tmp_path/'models').glob('*_last.pth')),map_location='cpu')
    assert checkpoint['epoch']==1
    assert checkpoint['selection_policy']=='last_epoch'
    assert checkpoint['vl_loss'] is None
    assert all(g['lr']==cfg['training']['optimizer']['params']['lr']
               for g in checkpoint['optimizer']['param_groups'])
