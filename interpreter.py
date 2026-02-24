from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lexer import LexerError, tokenize
from parser import (
    AssignStmt,
    BinaryExpr,
    BlockExpr,
    InterpolatedStringExpr,
    MethodDefStmt,
    NameExpr,
    RangeExpr,
    TypeExtendsStmt,
    BoolExpr,
    CallExpr,
    CheckExpr,
    Expr,
    ExprStmt,
    ForStmt,
    FunctionExpr,
    IfExpr,
    ImportStmt,
    IndexExpr,
    ListExpr,
    MemberExpr,
    NumberExpr,
    ParseError,
    PipeExprStage,
    PipeFilterStage,
    PipeMapStage,
    PipeOperatorStage,
    PipeSortStage,
    PipelineExpr,
    Program,
    QueryFindExpr,
    QueryFromExpr,
    RecordExpr,
    StringExpr,
    UnaryExpr,
    WhileStmt,
    parse,
)


class RuntimeErrorVal(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"RuntimeError at {line}:{column} - {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass
class GCBox:
    value: Any
    refs: int = 1

    def retain(self) -> None:
        self.refs += 1

    def release(self) -> None:
        self.refs -= 1
        if self.refs <= 0:
            self.value = None


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.values: Dict[str, GCBox] = {}

    def define(self, name: str, value: Any) -> None:
        if name in self.values:
            self.values[name].release()
        self.values[name] = GCBox(value, 1)

    def set(self, name: str, value: Any) -> None:
        if name in self.values:
            self.values[name].release()
            self.values[name] = GCBox(value, 1)
            return
        if self.parent:
            self.parent.set(name, value)
            return
        self.values[name] = GCBox(value, 1)

    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name].value
        if self.parent:
            return self.parent.get(name)
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.parent:
            out.update(self.parent.to_dict())
        for k, v in self.values.items():
            out[k] = v.value
        return out

    def close(self) -> None:
        for box in self.values.values():
            box.release()
        self.values.clear()


@dataclass
class UserFunction:
    params: List[tuple]
    body: Expr
    closure: Environment
    name: str = "<lambda>"

    def __call__(self, interpreter: "Interpreter", args: List[Any], named_args: Dict[str, Any]) -> Any:
        if named_args:
            raise ValueError("Named arguments are not supported for function literals")
        param_names = [p[0] for p in self.params]
        defaults = [p[1] for p in self.params]
        if len(args) > len(self.params):
            raise ValueError(f"{self.name} expects at most {len(self.params)} args, got {len(args)}")
        filled: List[Any] = list(args)
        for i in range(len(args), len(self.params)):
            if defaults[i] is not None:
                filled.append(interpreter.eval_expr(defaults[i], self.closure))
            else:
                raise ValueError(f"{self.name} missing required argument '{param_names[i]}'")
        call_env = Environment(self.closure)
        try:
            for idx, param in enumerate(param_names):
                call_env.define(param, filled[idx])
            return interpreter.eval_expr(self.body, call_env)
        finally:
            call_env.close()


@dataclass
class BuiltinFunction:
    name: str
    fn: Callable[..., Any]

    def __call__(self, interpreter: "Interpreter", args: List[Any], named_args: Dict[str, Any]) -> Any:
        if named_args:
            return self.fn(*args, **named_args)
        return self.fn(*args)


def val_to_py(val: Any) -> Any:
    """Convert Val value to Python-native for passing to Python APIs."""
    if isinstance(val, PyRef):
        return val.obj
    if isinstance(val, list):
        return [val_to_py(x) for x in val]
    if isinstance(val, dict):
        return {k: val_to_py(v) for k, v in val.items() if k not in ("__type__", "__struct__")}
    return val


def py_to_val(py: Any) -> Any:
    """Convert Python value to Val; wrap ndarray/tensor/modules/callables in PyRef."""
    if py is None or isinstance(py, (bool, int, float, str)):
        return py
    if isinstance(py, (list, tuple)):
        return [py_to_val(x) for x in py]
    if isinstance(py, dict):
        return {k: py_to_val(v) for k, v in py.items()}
    return PyRef(py)


@dataclass
class PyRef:
    """Wrapper for Python objects to enable Val interop."""

    obj: Any

    def __call__(self, interpreter: "Interpreter", args: List[Any], named_args: Dict[str, Any]) -> Any:
        py_args = [val_to_py(a) for a in args]
        py_named = {k: val_to_py(v) for k, v in named_args.items()}
        try:
            result = self.obj(*py_args, **py_named)
            return py_to_val(result)
        except Exception as err:
            raise RuntimeErrorVal(str(err), 1, 1)


@dataclass
class BoundMethod:
    instance: Any
    method: "UserFunction"

    def __call__(self, interpreter: "Interpreter", args: List[Any], named_args: Dict[str, Any]) -> Any:
        return self.method(interpreter, [self.instance] + args, named_args)


@dataclass
class StructType:
    name: str
    fields: Dict[str, str]
    methods: Dict[str, "UserFunction"]
    parent: Optional["StructType"] = None

    def _all_fields(self) -> Dict[str, str]:
        if self.parent:
            return {**self.parent._all_fields(), **self.fields}
        return dict(self.fields)

    def _get_method(self, name: str) -> Optional["UserFunction"]:
        if name in self.methods:
            return self.methods[name]
        if self.parent:
            return self.parent._get_method(name)
        return None

    def __call__(self, interpreter: "Interpreter", args: List[Any], named_args: Dict[str, Any]) -> Any:
        if args:
            raise ValueError(f"{self.name} expects named arguments")
        all_fields = self._all_fields()
        instance = {"__type__": self.name, "__struct__": self}
        for field in all_fields:
            if field not in named_args:
                raise ValueError(f"Missing field '{field}' for {self.name}")
            instance[field] = named_args[field]
        return instance


class Interpreter:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.std_root = self.project_root / "std"
        self.loaded_modules: Dict[str, bool] = {}
        self._module_cache: Dict[str, Dict[str, Any]] = {}
        self.global_env = Environment()
        self._install_builtins(self.global_env)

    def run_source(self, source: str, filename: str = "<repl>") -> Any:
        try:
            tokens = tokenize(source)
            program = parse(tokens)
            return self.eval_program(program, self.global_env, filename)
        except (LexerError, ParseError, RuntimeErrorVal) as err:
            raise err

    def eval_program(self, program: Program, env: Environment, filename: str) -> Any:
        result = None
        for stmt in program.statements:
            result = self.eval_stmt(stmt, env, filename)
        return result

    def eval_stmt(self, stmt: Any, env: Environment, filename: str) -> Any:
        if isinstance(stmt, ImportStmt):
            path = stmt.module_path
            if path.startswith("python:"):
                mod_name = path[7:].strip()
                if not mod_name:
                    raise RuntimeErrorVal("import 'python:' requires module name", stmt.line, stmt.column)
                try:
                    import importlib
                    mod = importlib.import_module(mod_name)
                    binding = stmt.alias or mod_name.split(".")[-1]
                    env.define(binding, PyRef(mod))
                except ImportError as err:
                    raise RuntimeErrorVal(f"Cannot import Python module '{mod_name}': {err}", stmt.line, stmt.column)
                return None
            self.import_module(path, filename, env, stmt.alias)
            return None
        if isinstance(stmt, TypeExtendsStmt):
            parent = env.get(stmt.parent_name)
            if not isinstance(parent, StructType):
                raise RuntimeErrorVal(f"'{stmt.parent_name}' is not a type", stmt.line, stmt.column)
            extra_fields: Dict[str, str] = {}
            for k, v in stmt.extra_fields.items():
                if not isinstance(v, NameExpr):
                    raise RuntimeErrorVal(f"Extended type field '{k}' must be a type", stmt.line, stmt.column)
                extra_fields[k] = v.name
            child = StructType(name=stmt.type_name, fields=extra_fields, methods={}, parent=parent)
            env.define(stmt.type_name, child)
            return None

        if isinstance(stmt, MethodDefStmt):
            type_val = env.get(stmt.type_name)
            if not isinstance(type_val, StructType):
                raise RuntimeErrorVal(f"'{stmt.type_name}' is not a type", stmt.line, stmt.column)
            method_fn = UserFunction([(p, None) for p in stmt.params], stmt.body, env)
            type_val.methods[stmt.method_name] = method_fn
            return None

        if isinstance(stmt, AssignStmt):
            struct_type = self._maybe_struct_type(stmt.name, stmt.value)
            if struct_type is not None:
                value = struct_type
            else:
                value = self.eval_expr(stmt.value, env)
            env.define(stmt.name, value)
            return value
        if isinstance(stmt, ExprStmt):
            return self.eval_expr(stmt.expr, env)
        if isinstance(stmt, ForStmt):
            iterable = self.eval_expr(stmt.iterable, env)
            result = None
            for item in iterable:
                env.define(stmt.var_name, item)
                result = self.eval_expr(stmt.body, env)
            return result
        if isinstance(stmt, WhileStmt):
            result = None
            while self._truthy(self.eval_expr(stmt.condition, env)):
                result = self.eval_expr(stmt.body, env)
            return result
        raise RuntimeErrorVal(f"Unsupported statement: {type(stmt).__name__}", stmt.line, stmt.column)

    def eval_expr(self, expr: Expr, env: Environment) -> Any:
        if isinstance(expr, NumberExpr):
            return expr.value
        if isinstance(expr, StringExpr):
            return expr.value
        if isinstance(expr, InterpolatedStringExpr):
            result_parts: List[str] = []
            for literal, sub_expr in expr.parts:
                if sub_expr is not None:
                    result_parts.append(str(self.eval_expr(sub_expr, env)))
                else:
                    result_parts.append(literal)
            return "".join(result_parts)
        if isinstance(expr, BoolExpr):
            return expr.value
        if isinstance(expr, NameExpr):
            try:
                return env.get(expr.name)
            except KeyError:
                msg = f"Undefined variable '{expr.name}'"
                suggestion = self._suggest_name(expr.name, env)
                if suggestion:
                    msg += f" (did you mean '{suggestion}'?)"
                raise RuntimeErrorVal(msg, expr.line, expr.column)
        if isinstance(expr, ListExpr):
            return [self.eval_expr(item, env) for item in expr.elements]
        if isinstance(expr, RecordExpr):
            return {k: self.eval_expr(v, env) for k, v in expr.entries.items()}
        if isinstance(expr, BlockExpr):
            result = None
            for stmt in expr.statements:
                result = self.eval_stmt(stmt, env, "<block>")
            return result
        if isinstance(expr, FunctionExpr):
            return UserFunction(expr.params, expr.body, env)
        if isinstance(expr, UnaryExpr):
            value = self.eval_expr(expr.expr, env)
            if expr.op == "-":
                return -value
            if expr.op == "+":
                return +value
            if expr.op == "not":
                return not self._truthy(value)
            raise RuntimeErrorVal(f"Unsupported unary operator {expr.op}", expr.line, expr.column)
        if isinstance(expr, BinaryExpr):
            left = self.eval_expr(expr.left, env)
            if expr.op == "and":
                return self._truthy(left) and self._truthy(self.eval_expr(expr.right, env))
            if expr.op == "or":
                return self._truthy(left) or self._truthy(self.eval_expr(expr.right, env))
            right = self.eval_expr(expr.right, env)
            return self._binary(left, expr.op, right, expr.line, expr.column)
        if isinstance(expr, IfExpr):
            if self._truthy(self.eval_expr(expr.condition, env)):
                return self.eval_expr(expr.then_expr, env)
            return self.eval_expr(expr.else_expr, env)
        if isinstance(expr, CheckExpr):
            subject = self.eval_expr(expr.subject, env)
            for cond_expr, out_expr in expr.branches:
                if self._truthy(self._eval_with_subject(cond_expr, subject, env)):
                    return self.eval_expr(out_expr, env)
            return self.eval_expr(expr.default_expr, env)
        if isinstance(expr, CallExpr):
            callee = self.eval_expr(expr.callee, env)
            args = [self.eval_expr(a, env) for a in expr.args]
            named = {k: self.eval_expr(v, env) for k, v in expr.named_args.items()}
            return self._call(callee, args, named, expr.line, expr.column)
        if isinstance(expr, MemberExpr):
            obj = self.eval_expr(expr.object_expr, env)
            if isinstance(obj, PyRef):
                try:
                    result = getattr(obj.obj, expr.name)
                    return py_to_val(result)
                except AttributeError as err:
                    raise RuntimeErrorVal(str(err), expr.line, expr.column)
            if isinstance(obj, dict):
                if expr.name in obj:
                    return obj[expr.name]
                if "__struct__" in obj:
                    struct = obj["__struct__"]
                    method = struct._get_method(expr.name)
                    if method is not None:
                        return BoundMethod(obj, method)
                raise RuntimeErrorVal(f"Field '{expr.name}' not found", expr.line, expr.column)
            if hasattr(obj, expr.name):
                return getattr(obj, expr.name)
            raise RuntimeErrorVal(f"Cannot access member '{expr.name}'", expr.line, expr.column)
        if isinstance(expr, IndexExpr):
            obj = self.eval_expr(expr.object_expr, env)
            idx = self.eval_expr(expr.index_expr, env)
            if isinstance(obj, PyRef):
                try:
                    result = obj.obj[val_to_py(idx)]
                    return py_to_val(result)
                except Exception as err:
                    raise RuntimeErrorVal(str(err), expr.line, expr.column)
            try:
                return obj[idx]
            except Exception as err:
                raise RuntimeErrorVal(str(err), expr.line, expr.column)
        if isinstance(expr, RangeExpr):
            start = self.eval_expr(expr.start, env)
            end = self.eval_expr(expr.end, env)
            return list(range(int(start), int(end) + 1))
        if isinstance(expr, PipelineExpr):
            return self._eval_pipeline(expr, env)
        if isinstance(expr, QueryFromExpr):
            return self._eval_query_from(expr, env)
        if isinstance(expr, QueryFindExpr):
            return self._eval_query_find(expr, env)
        raise RuntimeErrorVal(f"Unsupported expression: {type(expr).__name__}", expr.line, expr.column)

    def _eval_pipeline(self, expr: PipelineExpr, env: Environment) -> Any:
        current = self.eval_expr(expr.left, env)
        for stage in expr.stages:
            if isinstance(stage, PipeOperatorStage):
                rhs = self.eval_expr(stage.rhs, env)
                current = self._binary(current, stage.op, rhs, stage.line, stage.column)
                continue
            if isinstance(stage, PipeMapStage):
                if not isinstance(current, list):
                    raise RuntimeErrorVal("map pipeline stage expects list", stage.line, stage.column)
                current = [self._eval_with_it(stage.mapper, item, env) for item in current]
                continue
            if isinstance(stage, PipeFilterStage):
                if not isinstance(current, list):
                    raise RuntimeErrorVal("filter pipeline stage expects list", stage.line, stage.column)
                current = [item for item in current if self._truthy(self._eval_with_it(stage.predicate, item, env))]
                continue
            if isinstance(stage, PipeSortStage):
                if not isinstance(current, list):
                    raise RuntimeErrorVal("sort pipeline stage expects list", stage.line, stage.column)
                if stage.key_expr is None:
                    current = sorted(current)
                else:
                    current = sorted(current, key=lambda v: self._eval_with_it(stage.key_expr, v, env))
                continue
            if isinstance(stage, PipeExprStage):
                current = self._eval_pipe_expr_stage(current, stage.expr, env, stage.line, stage.column)
                continue
            raise RuntimeErrorVal("Unknown pipeline stage", stage.line, stage.column)
        return current

    def _eval_pipe_expr_stage(self, current: Any, stage_expr: Expr, env: Environment, line: int, column: int) -> Any:
        if isinstance(stage_expr, NameExpr):
            callee = self.eval_expr(stage_expr, env)
            return self._call(callee, [current], {}, line, column)
        if isinstance(stage_expr, CallExpr):
            callee = self.eval_expr(stage_expr.callee, env)
            args = [current] + [self.eval_expr(arg, env) for arg in stage_expr.args]
            named = {k: self.eval_expr(v, env) for k, v in stage_expr.named_args.items()}
            return self._call(callee, args, named, line, column)
        stage_val = self.eval_expr(stage_expr, env)
        if callable(stage_val):
            return self._call(stage_val, [current], {}, line, column)
        return stage_val

    def _eval_query_from(self, expr: QueryFromExpr, env: Environment) -> Any:
        source = self.eval_expr(expr.source, env)
        if not isinstance(source, list):
            raise RuntimeErrorVal("from query source must be a list", expr.line, expr.column)
        out: List[Any] = []
        for item in source:
            if expr.where_expr is not None and not self._truthy(self._eval_with_it(expr.where_expr, item, env)):
                continue
            out.append(self._eval_with_it(expr.select_expr, item, env))
        return out

    def _eval_query_find(self, expr: QueryFindExpr, env: Environment) -> Any:
        source = self.eval_expr(expr.source, env)
        if not isinstance(source, list):
            raise RuntimeErrorVal("find query source must be a list", expr.line, expr.column)
        out: List[Any] = []
        for item in source:
            sub_env = Environment(env)
            try:
                sub_env.define("it", item)
                sub_env.define(expr.item_name, item)
                if self._truthy(self.eval_expr(expr.where_expr, sub_env)):
                    out.append(item)
            finally:
                sub_env.close()
        return out

    def _eval_with_it(self, expression: Expr, item: Any, env: Environment) -> Any:
        sub_env = Environment(env)
        try:
            sub_env.define("it", item)
            return self.eval_expr(expression, sub_env)
        finally:
            sub_env.close()

    def _eval_with_subject(self, expression: Expr, subject: Any, env: Environment) -> Any:
        sub_env = Environment(env)
        try:
            sub_env.define("it", subject)
            return self.eval_expr(expression, sub_env)
        finally:
            sub_env.close()

    def _call(self, callee: Any, args: List[Any], named: Dict[str, Any], line: int, column: int) -> Any:
        try:
            if isinstance(callee, (UserFunction, BuiltinFunction, StructType, BoundMethod, PyRef)):
                return callee(self, args, named)
            if callable(callee):
                return callee(*args, **named)
        except Exception as err:
            raise RuntimeErrorVal(str(err), line, column)
        raise RuntimeErrorVal("Attempted to call non-function value", line, column)

    def _binary(self, left: Any, op: str, right: Any, line: int, column: int) -> Any:
        if isinstance(left, PyRef) or isinstance(right, PyRef):
            py_left = val_to_py(left)
            py_right = val_to_py(right)
            op_map = {
                "+": "__add__", "-": "__sub__", "*": "__mul__", "/": "__truediv__", "%": "__mod__",
                ">": "__gt__", ">=": "__ge__", "<": "__lt__", "<=": "__le__",
                "==": "__eq__", "!=": "__ne__",
            }
            if op in op_map:
                method = op_map[op]
                try:
                    result = getattr(py_left, method)(py_right)
                    if result is NotImplemented and py_left is not py_right:
                        result = getattr(py_right, "__r" + method[2:])(py_left)
                    if result is not NotImplemented:
                        return py_to_val(result)
                except (TypeError, AttributeError):
                    pass
            if op == "in":
                try:
                    return py_right.__contains__(py_left)
                except (TypeError, AttributeError):
                    pass
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "%":
                return left % right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "in":
                return left in right
        except Exception as err:
            raise RuntimeErrorVal(str(err), line, column)
        raise RuntimeErrorVal(f"Unsupported operator '{op}'", line, column)

    def _truthy(self, value: Any) -> bool:
        return bool(value)

    def _suggest_name(self, name: str, env: Environment) -> Optional[str]:
        def names(e: Environment) -> set:
            out = set(e.values.keys())
            if e.parent:
                out |= names(e.parent)
            return out
        all_names = names(env)
        if not all_names:
            return None
        best, best_dist = None, float("inf")
        for n in all_names:
            dist = sum(a != b for a, b in zip(name, n)) + abs(len(name) - len(n))
            if dist < best_dist and dist <= 2:
                best, best_dist = n, dist
        return best

    def import_module(self, module_path: str, current_file: str, env: Optional[Environment] = None, alias: Optional[str] = None) -> None:
        resolved = self._resolve_module_path(module_path, current_file)
        key = str(resolved)
        target_env = env or self.global_env
        if key in self.loaded_modules:
            if alias:
                mod = self._module_cache.get(key)
                if mod is not None:
                    target_env.define(alias, mod)
            return
        self.loaded_modules[key] = True
        source = resolved.read_text(encoding="utf-8")
        if alias:
            module_env = Environment(self.global_env)
            try:
                tokens = tokenize(source)
                program = parse(tokens)
                self.eval_program(program, module_env, key)
                mod = module_env.to_dict()
                self._module_cache[key] = mod
                target_env.define(alias, mod)
            finally:
                module_env.close()
        else:
            self.run_source(source, filename=key)

    def _resolve_module_path(self, module_path: str, current_file: str) -> Path:
        path_text = module_path
        if not path_text.endswith(".val"):
            path_text = f"{path_text}.val"

        candidate = self.std_root / path_text
        if candidate.exists():
            return candidate.resolve()

        current = Path(current_file)
        if current.exists():
            local = current.parent / path_text
            if local.exists():
                return local.resolve()

        local = self.project_root / path_text
        if local.exists():
            return local.resolve()

        raise RuntimeErrorVal(f"Cannot import module '{module_path}'", 1, 1)

    def _maybe_struct_type(self, name: str, value_expr: Expr) -> Optional[StructType]:
        if not isinstance(value_expr, RecordExpr):
            return None
        fields: Dict[str, str] = {}
        for key, expr in value_expr.entries.items():
            if not isinstance(expr, NameExpr):
                return None
            fields[key] = expr.name
        return StructType(name=name, fields=fields, methods={}, parent=None)

    def _install_builtins(self, env: Environment) -> None:
        def _type_name(value: Any) -> str:
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, (int, float)):
                return "number"
            if isinstance(value, str):
                return "string"
            if isinstance(value, list):
                return "list"
            if isinstance(value, dict):
                if "__type__" in value:
                    return value["__type__"]
                return "map"
            if isinstance(value, UserFunction):
                return "function"
            return type(value).__name__

        def _py_import(name: str) -> PyRef:
            try:
                import importlib
                mod = importlib.import_module(name)
                return PyRef(mod)
            except ImportError as err:
                raise ValueError(f"Cannot import Python module '{name}': {err}")

        def _array(*data: Any) -> PyRef:
            try:
                import importlib
                np = importlib.import_module("numpy")
                if len(data) == 1:
                    return PyRef(np.array(val_to_py(data[0])))
                return PyRef(np.array([val_to_py(x) for x in data]))
            except ImportError:
                raise ValueError("array() requires numpy; install with: pip install numpy")

        def _py_call(obj: Any, method: str, *args: Any) -> Any:
            if isinstance(obj, PyRef):
                fn = getattr(obj.obj, method)
                py_args = [val_to_py(a) for a in args]
                return py_to_val(fn(*py_args))
            raise ValueError("py_call expects a PyRef (Python object)")

        def _print(*args: Any) -> None:
            out = []
            for a in args:
                out.append(str(a.obj) if isinstance(a, PyRef) else a)
            print(*out)

        builtins: Dict[str, Callable[..., Any]] = {
            "print": _print,
            "input": input,
            "len": len,
            "type": _type_name,
            "split": lambda s, sep=" ": str(s).split(sep),
            "join": lambda items, sep="": str(sep).join(str(x) for x in items),
            "read": lambda path: Path(path).read_text(encoding="utf-8"),
            "write": lambda path, content: Path(path).write_text(str(content), encoding="utf-8"),
            "range": lambda start, stop=None, step=1: list(range(start, stop, step)) if stop is not None else list(range(start)),
            "each": lambda items: list(items),
            "map": lambda items, fn: [fn(x) for x in items],
            "filter": lambda items, fn: [x for x in items if fn(x)],
            "reduce": self._builtin_reduce,
            "sort": lambda items: sorted(items),
            "reverse": lambda items: list(reversed(items)),
            "keys": lambda m: list(m.keys()),
            "values": lambda m: list(m.values()),
            "sum": lambda items: sum(items),
            "not_null": lambda x: x is not None,
            "py_import": _py_import,
            "py_call": _py_call,
            "array": _array,
        }
        for name, fn in builtins.items():
            env.define(name, BuiltinFunction(name, fn))

    def _builtin_reduce(self, items: List[Any], fn: Any, initial: Any = None) -> Any:
        iterator = iter(items)
        if initial is None:
            try:
                acc = next(iterator)
            except StopIteration:
                raise ValueError("reduce of empty sequence with no initial value")
        else:
            acc = initial
        for item in iterator:
            if isinstance(fn, (UserFunction, BuiltinFunction)):
                acc = fn(self, [acc, item], {})
            elif callable(fn):
                acc = fn(acc, item)
            else:
                raise ValueError("reduce expects callable function")
        return acc

