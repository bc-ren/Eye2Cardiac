"""Canonical names and legacy dispatch, without data or framework imports.

Only class/function declarations are compiled; a stand-in superclass avoids
loading the private model dependencies. These are naming/dispatch tests, not
neural-network forward tests or checkpoint-weight loading tests.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def declarations(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in names:
            nodes.append(node)
            found.add(name)
    if found != names:
        raise AssertionError(f'Missing naming declarations: {names - found}')
    env = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), env)
    return env


class ModelNamingTests(unittest.TestCase):
    def test_cardiacae_registry_and_legacy_dispatch(self):
        env = declarations('src/teacher/model.py',
                           {'MODEL_REGISTRY', 'register_model', 'make_model', 'CardiacAE', 'LVMyoTeacher'},
                           {'nn': SimpleNamespace(Module=object)})
        cls = env['CardiacAE']
        self.assertEqual(cls.__name__, 'CardiacAE')
        self.assertIs(env['LVMyoTeacher'], cls)
        for key in ('CardiacAE', 'lv_myo_nonode_core64_detail32_v1'):
            self.assertIs(env['MODEL_REGISTRY'][key], cls)
            # Test factory routing only; constructing a real model needs private topology.
            with patch.object(cls, '__init__', lambda self, cfg: None):
                self.assertIs(type(env['make_model']({'architecture_id': key})), cls)

    def test_reticardiac_public_alias(self):
        env = declarations('src/student/student.py', {'RetiCardiac', 'Student'},
                           {'base': SimpleNamespace(RetiCardiacBase=object)})
        self.assertEqual(env['RetiCardiac'].__name__, 'RetiCardiac')
        self.assertIs(env['Student'], env['RetiCardiac'])

    def test_reticardiac_base_alias(self):
        env = declarations('src/student/reference/student.py', {'RetiCardiacBase', 'Student'},
                           {'CardiacQueryStudent': object})
        self.assertIs(env['Student'], env['RetiCardiacBase'])

    def test_factories_use_canonical_names(self):
        for relative, expected in [('src/student/student.py', 'RetiCardiac'),
                                   ('src/student/reference/student.py', 'RetiCardiacBase')]:
            tree = ast.parse((ROOT / relative).read_text())
            factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_models')
            calls = {n.func.id for n in ast.walk(factory) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            self.assertIn(expected, calls)
            self.assertNotIn('Student', calls)

    def test_config_model_names(self):
        self.assertIn('architecture_id: CardiacAE\n', (ROOT / 'configs/teacher.yaml').read_text())
        for name in ('student.json', 'student_model.json'):
            config = json.loads((ROOT / 'configs' / name).read_text())
            self.assertEqual(config['model_name'], 'RetiCardiac')
            self.assertEqual(config['teacher_name'], 'CardiacAE')
        config = json.loads((ROOT / 'configs/downstream.json').read_text())
        self.assertEqual(config['feature_model_name'], 'RetiCardiac')


if __name__ == '__main__':
    unittest.main()
