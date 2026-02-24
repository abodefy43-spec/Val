from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from lexer import Token


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"ParseError at {token.line}:{token.column} - {message}")
        self.message = message
        self.token = token


@dataclass
class Node:
    line: int
    column: int


@dataclass
class Program(Node):
    statements: List["Stmt"]


@dataclass
class Stmt(Node):
    pass


@dataclass
class ImportStmt(Stmt):
    module_path: str
    alias: Optional[str] = None


@dataclass
class AssignStmt(Stmt):
    name: str
    value: "Expr"


@dataclass
class MethodDefStmt(Stmt):
    type_name: str
    method_name: str
    params: List[str]
    body: Expr


@dataclass
class TypeExtendsStmt(Stmt):
    type_name: str
    parent_name: str
    extra_fields: Dict[str, Expr]


@dataclass
class ExprStmt(Stmt):
    expr: "Expr"


@dataclass
class Expr(Node):
    pass


@dataclass
class NameExpr(Expr):
    name: str


@dataclass
class NumberExpr(Expr):
    value: float


@dataclass
class StringExpr(Expr):
    value: str


@dataclass
class InterpolatedStringExpr(Expr):
    parts: List[Tuple[str, Optional[Expr]]]


@dataclass
class BoolExpr(Expr):
    value: bool


@dataclass
class ListExpr(Expr):
    elements: List[Expr]


@dataclass
class RecordExpr(Expr):
    entries: Dict[str, Expr]


@dataclass
class FunctionExpr(Expr):
    params: List[Tuple[str, Optional[Expr]]]
    body: Expr


@dataclass
class CallExpr(Expr):
    callee: Expr
    args: List[Expr]
    named_args: Dict[str, Expr]


@dataclass
class MemberExpr(Expr):
    object_expr: Expr
    name: str


@dataclass
class IndexExpr(Expr):
    object_expr: Expr
    index_expr: Expr


@dataclass
class UnaryExpr(Expr):
    op: str
    expr: Expr


@dataclass
class BinaryExpr(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass
class IfExpr(Expr):
    condition: Expr
    then_expr: Expr
    else_expr: Expr


@dataclass
class CheckExpr(Expr):
    subject: Expr
    branches: List[Tuple[Expr, Expr]]
    default_expr: Expr


@dataclass
class QueryFromExpr(Expr):
    source: Expr
    where_expr: Optional[Expr]
    select_expr: Expr


@dataclass
class QueryFindExpr(Expr):
    item_name: str
    source: Expr
    where_expr: Expr


@dataclass
class PipeStage(Node):
    pass


@dataclass
class PipeExprStage(PipeStage):
    expr: Expr


@dataclass
class PipeOperatorStage(PipeStage):
    op: str
    rhs: Expr


@dataclass
class PipeMapStage(PipeStage):
    mapper: Expr


@dataclass
class PipeFilterStage(PipeStage):
    predicate: Expr


@dataclass
class PipeSortStage(PipeStage):
    key_expr: Optional[Expr]


@dataclass
class PipelineExpr(Expr):
    left: Expr
    stages: List[PipeStage]


@dataclass
class RangeExpr(Expr):
    start: Expr
    end: Expr


@dataclass
class BlockExpr(Expr):
    statements: List["Stmt"]


@dataclass
class ForStmt(Stmt):
    var_name: str
    iterable: Expr
    body: Expr


@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: Expr


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = 0

    def parse(self) -> Program:
        statements: List[Stmt] = []
        self._skip_newlines()
        while not self._is("EOF"):
            statements.append(self._parse_statement())
            self._skip_newlines()
        start = self.tokens[0] if self.tokens else Token("EOF", "", 1, 1)
        return Program(start.line, start.column, statements)

    def _parse_statement(self) -> Stmt:
        if self._is_keyword("import"):
            tok = self._advance()
            module = self._consume("STRING", "Expected import path string")
            alias: Optional[str] = None
            if self._is_keyword("as"):
                self._advance()
                alias = self._consume("NAME", "Expected alias name after 'as'").value
            return ImportStmt(tok.line, tok.column, module.value, alias)

        if self._is("NAME") and self._peek_next().kind == "KEYWORD" and self.tokens[self.index + 1].value == "extends":
            name_tok = self._advance()
            self._advance()
            parent_tok = self._consume("NAME", "Expected parent type name")
            self._consume("LBRACE", "Expected '{' for extended type fields")
            extra = self._parse_record(name_tok.line, name_tok.column)
            return TypeExtendsStmt(name_tok.line, name_tok.column, name_tok.value, parent_tok.value, extra.entries)

        if self._is_keyword("for"):
            tok = self._advance()
            var_name = self._consume("NAME", "Expected variable name after for").value
            self._consume_keyword("in", "Expected 'in' after for variable")
            iterable = self._parse_expression()
            self._skip_newlines()
            if self._is_keyword("do"):
                self._advance()
                body = self._parse_expression()
            elif self._match("LBRACE"):
                body = self._parse_block(tok.line, tok.column)
            else:
                raise ParseError("Expected 'do' or '{' for for loop body", self._peek())
            return ForStmt(tok.line, tok.column, var_name, iterable, body)

        if self._is_keyword("while"):
            tok = self._advance()
            condition = self._parse_expression()
            self._skip_newlines()
            if self._is_keyword("do"):
                self._advance()
                body = self._parse_expression()
            elif self._match("LBRACE"):
                body = self._parse_block(tok.line, tok.column)
            else:
                raise ParseError("Expected 'do' or '{' for while loop body", self._peek())
            return WhileStmt(tok.line, tok.column, condition, body)

        if self._is("NAME"):
            type_tok = self._peek()
            tok1 = self.tokens[self.index + 1] if self.index + 1 < len(self.tokens) else None
            tok2 = self.tokens[self.index + 2] if self.index + 2 < len(self.tokens) else None
            tok3 = self.tokens[self.index + 3] if self.index + 3 < len(self.tokens) else None
            if (tok1 and tok1.kind == "DOT" and tok2 and tok2.kind == "NAME" and
                    tok3 and tok3.kind == "EQ"):
                self._advance()
                self._advance()
                method_tok = self._advance()
                self._advance()
                self._consume("LPAREN", "Expected '(' for method parameters")
                params: List[str] = []
                if not self._is("RPAREN"):
                    while True:
                        params.append(self._consume("NAME", "Expected parameter name").value)
                        if not self._match("COMMA"):
                            break
                self._consume("RPAREN", "Expected ')'")
                self._consume("ARROW", "Expected '=>'")
                body = self._parse_block_or_expr()
                return MethodDefStmt(type_tok.line, type_tok.column, type_tok.value,
                                    method_tok.value, params, body)

        if self._is("NAME") and self._peek_next().kind == "EQ":
            name_tok = self._advance()
            self._advance()
            value = self._parse_expression()
            return AssignStmt(name_tok.line, name_tok.column, name_tok.value, value)

        expr = self._parse_expression()
        return ExprStmt(expr.line, expr.column, expr)

    def _parse_expression(self, allow_comma: bool = True) -> Expr:
        expr = self._parse_lambda()
        if not allow_comma or not self._is("COMMA"):
            return expr
        elements = [expr]
        while self._match("COMMA"):
            elements.append(self._parse_lambda())
        return ListExpr(expr.line, expr.column, elements)

    def _parse_lambda(self) -> Expr:
        if self._is("NAME") and self._peek_next().kind == "ARROW":
            param = self._advance()
            self._advance()
            body = self._parse_block_or_expr()
            return FunctionExpr(param.line, param.column, [(param.value, None)], body)

        checkpoint = self.index
        if self._match("LPAREN"):
            params: List[Tuple[str, Optional[Expr]]] = []
            valid = True
            if not self._is("RPAREN"):
                while True:
                    if not self._is("NAME"):
                        valid = False
                        break
                    name_tok = self._advance()
                    default: Optional[Expr] = None
                    if self._match("EQ"):
                        default = self._parse_expression(allow_comma=False)
                    params.append((name_tok.value, default))
                    if not self._match("COMMA"):
                        break
            if valid and self._match("RPAREN") and self._match("ARROW"):
                body = self._parse_block_or_expr()
                tok = self.tokens[checkpoint]
                return FunctionExpr(tok.line, tok.column, params, body)
            self.index = checkpoint

        return self._parse_pipeline()

    def _parse_pipeline(self) -> Expr:
        left = self._parse_or()
        stages: List[PipeStage] = []
        while True:
            self._skip_newlines()
            if not self._match("PIPE"):
                break
            stages.append(self._parse_pipe_stage())
        if stages:
            return PipelineExpr(left.line, left.column, left, stages)
        return left

    def _parse_pipe_stage(self) -> PipeStage:
        tok = self._peek()
        if tok.kind in {"PLUS", "MINUS", "STAR", "SLASH", "PERCENT", "GT", "GTE", "LT", "LTE", "EQEQ", "NEQ"}:
            op = self._advance().value
            rhs = self._parse_or()
            return PipeOperatorStage(tok.line, tok.column, op, rhs)

        if self._is("NAME"):
            name_tok = self._advance()
            name = name_tok.value
            if name == "map":
                mapper = self._parse_predicate_fragment()
                return PipeMapStage(name_tok.line, name_tok.column, mapper)
            if name == "filter":
                pred = self._parse_predicate_fragment()
                return PipeFilterStage(name_tok.line, name_tok.column, pred)
            if name == "sort" and self._is_keyword("by"):
                self._advance()
                key = self._parse_atomish()
                return PipeSortStage(name_tok.line, name_tok.column, key)
            if name == "sort":
                return PipeSortStage(name_tok.line, name_tok.column, None)
            expr: Expr = NameExpr(name_tok.line, name_tok.column, name)
            while True:
                if self._match("LPAREN"):
                    args, named = self._parse_call_args()
                    expr = CallExpr(expr.line, expr.column, expr, args, named)
                    continue
                if self._match("DOT"):
                    name_tok2 = self._consume("NAME", "Expected member name after '.'")
                    expr = MemberExpr(expr.line, expr.column, expr, name_tok2.value)
                    continue
                if self._match("LBRACKET"):
                    idx = self._parse_expression()
                    self._consume("RBRACKET", "Expected ']'")
                    expr = IndexExpr(expr.line, expr.column, expr, idx)
                    continue
                break
            return PipeExprStage(name_tok.line, name_tok.column, expr)

        expr = self._parse_or()
        return PipeExprStage(expr.line, expr.column, expr)

    def _parse_predicate_fragment(self) -> Expr:
        tok = self._peek()
        if tok.kind in {"PLUS", "MINUS", "STAR", "SLASH", "PERCENT", "GT", "GTE", "LT", "LTE", "EQEQ", "NEQ"}:
            op = self._advance().value
            right = self._parse_or()
            it = NameExpr(tok.line, tok.column, "it")
            return BinaryExpr(tok.line, tok.column, it, op, right)
        return self._parse_or()

    def _parse_or(self) -> Expr:
        expr = self._parse_and()
        while self._is_keyword("or"):
            tok = self._advance()
            right = self._parse_and()
            expr = BinaryExpr(tok.line, tok.column, expr, "or", right)
        return expr

    def _parse_and(self) -> Expr:
        expr = self._parse_equality()
        while self._is_keyword("and"):
            tok = self._advance()
            right = self._parse_equality()
            expr = BinaryExpr(tok.line, tok.column, expr, "and", right)
        return expr

    def _parse_equality(self) -> Expr:
        expr = self._parse_membership()
        while self._is("EQEQ") or self._is("NEQ"):
            tok = self._advance()
            right = self._parse_membership()
            expr = BinaryExpr(tok.line, tok.column, expr, tok.value, right)
        return expr

    def _parse_membership(self) -> Expr:
        expr = self._parse_comparison()
        while self._is_keyword("in"):
            tok = self._advance()
            right = self._parse_comparison()
            expr = BinaryExpr(tok.line, tok.column, expr, "in", right)
        return expr

    def _parse_comparison(self) -> Expr:
        expr = self._parse_term()
        while self._is("GT") or self._is("GTE") or self._is("LT") or self._is("LTE"):
            tok = self._advance()
            right = self._parse_membership()
            expr = BinaryExpr(tok.line, tok.column, expr, tok.value, right)
        return expr

    def _parse_term(self) -> Expr:
        expr = self._parse_factor()
        while self._is("PLUS") or self._is("MINUS"):
            tok = self._advance()
            right = self._parse_factor()
            expr = BinaryExpr(tok.line, tok.column, expr, tok.value, right)
        return expr

    def _parse_factor(self) -> Expr:
        expr = self._parse_unary()
        while self._is("STAR") or self._is("SLASH") or self._is("PERCENT"):
            tok = self._advance()
            right = self._parse_unary()
            expr = BinaryExpr(tok.line, tok.column, expr, tok.value, right)
        return expr

    def _parse_unary(self) -> Expr:
        if self._is("MINUS") or self._is("PLUS"):
            tok = self._advance()
            right = self._parse_unary()
            return UnaryExpr(tok.line, tok.column, tok.value, right)
        if self._is_keyword("not"):
            tok = self._advance()
            right = self._parse_unary()
            return UnaryExpr(tok.line, tok.column, "not", right)
        return self._parse_call()

    def _parse_call(self) -> Expr:
        expr = self._parse_primary()
        if self._match("DOTDOT"):
            end = self._parse_primary()
            expr = RangeExpr(expr.line, expr.column, expr, end)
        while True:
            if self._match("LPAREN"):
                args, named = self._parse_call_args()
                expr = CallExpr(expr.line, expr.column, expr, args, named)
                continue
            if self._match("DOT"):
                name_tok = self._consume("NAME", "Expected member name")
                expr = MemberExpr(expr.line, expr.column, expr, name_tok.value)
                continue
            if self._match("LBRACKET"):
                idx = self._parse_expression()
                self._consume("RBRACKET", "Expected ']'")
                expr = IndexExpr(expr.line, expr.column, expr, idx)
                continue
            break
        return expr

    def _parse_call_args(self) -> Tuple[List[Expr], Dict[str, Expr]]:
        args: List[Expr] = []
        named: Dict[str, Expr] = {}
        if self._match("RPAREN"):
            return args, named
        while True:
            if self._is("NAME") and self._peek_next().kind == "COLON":
                key = self._advance().value
                self._advance()
                named[key] = self._parse_expression(allow_comma=False)
            else:
                args.append(self._parse_expression(allow_comma=False))
            if self._match("RPAREN"):
                break
            self._consume("COMMA", "Expected ',' between call arguments")
        return args, named

    def _parse_primary(self) -> Expr:
        tok = self._peek()
        if self._is("NUMBER"):
            num = self._advance()
            value = float(num.value)
            if value.is_integer():
                value = int(value)
            return NumberExpr(num.line, num.column, value)
        if self._is("STRING_PART") or self._is("INTERP_OPEN"):
            parts: List[Tuple[str, Optional[Expr]]] = []
            line, col = 1, 1
            while self._is("STRING_PART") or self._is("INTERP_OPEN"):
                if self._is("STRING_PART"):
                    p = self._advance()
                    if not parts:
                        line, col = p.line, p.column
                    parts.append((p.value, None))
                else:
                    op = self._advance()
                    if not parts:
                        line, col = op.line, op.column
                    expr = self._parse_expression()
                    self._consume("INTERP_CLOSE", "Expected '}' to close interpolation")
                    parts.append(("", expr))
            if len(parts) == 1 and parts[0][1] is None:
                return StringExpr(line, col, parts[0][0])
            return InterpolatedStringExpr(line, col, parts)
        if self._is("STRING"):
            s = self._advance()
            return StringExpr(s.line, s.column, s.value)
        if self._is("NAME"):
            name = self._advance()
            return NameExpr(name.line, name.column, name.value)
        if self._is_keyword("yes") or self._is_keyword("no"):
            b = self._advance()
            return BoolExpr(b.line, b.column, b.value == "yes")
        if self._is_keyword("if"):
            return self._parse_if()
        if self._is_keyword("check"):
            return self._parse_check()
        if self._is_keyword("from"):
            return self._parse_query_from()
        if self._is_keyword("find"):
            return self._parse_query_find()
        if self._match("LPAREN"):
            expr = self._parse_expression()
            self._consume("RPAREN", "Expected ')'")
            return expr
        if self._match("LBRACE"):
            self._skip_newlines()
            if self._match("RBRACE"):
                return RecordExpr(tok.line, tok.column, {})
            if self._is("NAME") and self._peek_next().kind == "COLON":
                return self._parse_record(tok.line, tok.column)
            return self._parse_block(tok.line, tok.column)
        raise ParseError(f"Unexpected token {tok.kind} ({tok.value})", tok)

    def _parse_atomish(self) -> Expr:
        if self._is("NAME"):
            tok = self._advance()
            return NameExpr(tok.line, tok.column, tok.value)
        if self._is("STRING"):
            tok = self._advance()
            return StringExpr(tok.line, tok.column, tok.value)
        if self._is("NUMBER"):
            tok = self._advance()
            val = float(tok.value)
            if val.is_integer():
                val = int(val)
            return NumberExpr(tok.line, tok.column, val)
        return self._parse_expression()

    def _parse_block_or_expr(self) -> Expr:
        if self._is("LBRACE"):
            tok = self._peek()
            self._advance()
            self._skip_newlines()
            if self._match("RBRACE"):
                raise ParseError("Empty block not allowed", self._peek())
            if self._is("NAME") and self._peek_next().kind == "COLON":
                return self._parse_record(tok.line, tok.column)
            return self._parse_block(tok.line, tok.column)
        return self._parse_expression()

    def _parse_block(self, line: int, column: int) -> BlockExpr:
        statements: List[Stmt] = []
        while True:
            statements.append(self._parse_statement())
            self._skip_newlines()
            if self._match("RBRACE"):
                break
            self._match("SEMICOLON")
            self._skip_newlines()
        if not statements:
            raise ParseError("Block must have at least one statement", self._peek())
        return BlockExpr(line, column, statements)

    def _parse_record(self, line: int, column: int) -> RecordExpr:
        entries: Dict[str, Expr] = {}
        self._skip_newlines()
        if self._match("RBRACE"):
            return RecordExpr(line, column, entries)
        while True:
            key = self._consume("NAME", "Expected record key")
            self._consume("COLON", "Expected ':' after key")
            value = self._parse_expression(allow_comma=False)
            entries[key.value] = value
            self._skip_newlines()
            if self._match("RBRACE"):
                break
            self._consume("COMMA", "Expected ',' between record entries")
            self._skip_newlines()
        return RecordExpr(line, column, entries)

    def _parse_if(self) -> IfExpr:
        tok = self._advance()
        cond = self._parse_expression()
        self._skip_newlines()
        self._consume_keyword("then", "Expected 'then' in if expression")
        then_expr = self._parse_expression()
        self._skip_newlines()
        self._consume_keyword("else", "Expected 'else' in if expression")
        else_expr = self._parse_expression()
        return IfExpr(tok.line, tok.column, cond, then_expr, else_expr)

    def _parse_check(self) -> CheckExpr:
        tok = self._advance()
        subject = self._parse_expression()
        self._skip_newlines()
        branches: List[Tuple[Expr, Expr]] = []
        while True:
            self._skip_newlines()
            if not self._is_keyword("when"):
                break
            when_tok = self._advance()
            condition = self._parse_predicate_fragment()
            if isinstance(condition, NameExpr):
                condition = BinaryExpr(when_tok.line, when_tok.column, subject, "==", condition)
            self._skip_newlines()
            self._consume_keyword("then", "Expected 'then' in check branch")
            out_expr = self._parse_expression()
            branches.append((condition, out_expr))
            self._skip_newlines()
        self._consume_keyword("default", "Expected 'default' in check expression")
        default_expr = self._parse_expression()
        return CheckExpr(tok.line, tok.column, subject, branches, default_expr)

    def _parse_query_from(self) -> QueryFromExpr:
        tok = self._advance()
        source = self._parse_expression()
        self._skip_newlines()
        where_expr: Optional[Expr] = None
        if self._is_keyword("where"):
            self._advance()
            where_expr = self._parse_or()
            self._skip_newlines()
        self._consume_keyword("select", "Expected 'select' in query")
        select_expr = self._parse_or()
        return QueryFromExpr(tok.line, tok.column, source, where_expr, select_expr)

    def _parse_query_find(self) -> QueryFindExpr:
        tok = self._advance()
        item_name = self._consume("NAME", "Expected item name after find").value
        self._skip_newlines()
        self._consume_keyword("in", "Expected 'in' after find item")
        source = self._parse_expression()
        self._skip_newlines()
        self._consume_keyword("where", "Expected 'where' in find query")
        where_expr = self._parse_or()
        return QueryFindExpr(tok.line, tok.column, item_name, source, where_expr)

    def _skip_newlines(self) -> None:
        while self._is("NEWLINE"):
            self._advance()

    def _consume(self, kind: str, message: str) -> Token:
        if self._is(kind):
            return self._advance()
        raise ParseError(message, self._peek())

    def _consume_keyword(self, word: str, message: str) -> Token:
        if self._is_keyword(word):
            return self._advance()
        raise ParseError(message, self._peek())

    def _match(self, kind: str) -> bool:
        if self._is(kind):
            self._advance()
            return True
        return False

    def _is(self, kind: str) -> bool:
        return self._peek().kind == kind

    def _is_keyword(self, word: str) -> bool:
        tok = self._peek()
        return tok.kind == "KEYWORD" and tok.value == word

    def _peek(self) -> Token:
        return self.tokens[self.index]

    def _peek_next(self) -> Token:
        if self.index + 1 >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.index + 1]

    def _advance(self) -> Token:
        tok = self.tokens[self.index]
        self.index += 1
        return tok


def parse(tokens: List[Token]) -> Program:
    return Parser(tokens).parse()

