"""Configure staging/safety only, using empty synthetic files (not scientific input validation)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ConfigureTests(unittest.TestCase):
    def test_escaped_path_template(self):
        spec=importlib.util.spec_from_file_location('configure_release',ROOT/'scripts/configure.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context={'RUN':'folder with "quote" and \\ slash'}
        result=module.resolve_template({'a':'${RUN}/student','nested':['${RUN}',35]},context)
        self.assertEqual(json.loads(json.dumps(result))['a'],context['RUN']+'/student')

    def test_stage_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();inputs=base/'inputs';inputs.mkdir()
            config=json.loads((ROOT/'configs/paths.example.json').read_text())
            file_keys={'RETIZERO_CHECKPOINT','TEACHER_CHECKPOINT','TEACHER_TOPOLOGY','FULL_TOPOLOGY','BASAL_RING_IDS','CLINICAL_CSV'}
            for k in config:
                path=inputs/k
                if k in file_keys:path.touch()
                else:path.mkdir()
                config[k]=str(path)
            prep=Path(config['STUDENT_PREPARED'])
            for name in ['same_day_train','same_day_validation','same_day_test','asynchronous_train','asynchronous_validation','asynchronous_test','joint_train','image_manifest']:
                (prep/(name+'_SERVER_ONLY.parquet')).touch()
            (prep/'fold_0').mkdir()
            for name in ['prior_raw.npz','scalers.npz','residual_whitening.npz','AUDIT.json']:(prep/'fold_0'/name).touch()
            for name in ['recorded_timecovered_SERVER_ONLY.parquet','historical_country_SERVER_ONLY.parquet']:
                (Path(config['DOWNSTREAM_PREPARED'])/name).touch()
            paths=base/'paths.json';paths.write_text(json.dumps(config))
            output=base/'new_run'
            cmd=[sys.executable,str(ROOT/'scripts/configure.py'),'--paths',str(paths),'--output',str(output)]
            checked=subprocess.run(cmd+['--check-only'],capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stderr)
            self.assertFalse(output.exists())
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            status=json.loads((output/'STATUS.json').read_text())
            self.assertEqual(status['status'],'CONFIGURED_NOT_TRAINED')
            self.assertFalse(status['training_started'])
            self.assertFalse((output/'student/PREFLIGHT_COMPLETED.json').exists())
            self.assertEqual((output/'student/prepared').resolve(),prep)
            settings=json.loads((output/'student/settings.json').read_text())
            self.assertEqual(settings['max_epochs'],35)
            self.assertEqual(settings['retina_trainable_blocks'],[20,21,22,23])
            self.assertEqual(settings['parent'],str(output/'student_base'))
            self.assertNotIn('${',(output/'teacher/config.yaml').read_text())
            repeated=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(repeated.returncode,0)
            self.assertIn('Refusing to overwrite',repeated.stderr)


if __name__=='__main__':unittest.main()
