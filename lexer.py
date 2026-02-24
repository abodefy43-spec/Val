from __future__ import annotations

from dataclasses import dataclass
from typing import List


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"LexerError at {line}:{column} - {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    column: int


KEYWORDS = {
    "from",
    "where",
    "select",
    "find",
    "in",
    "when",
    "then",
    "default",
    "if",
    "else",
    "check",
    "import",
    "by",
    "yes",
    "no",
    "and",
    "or",
    "not",
    "for",
    "do",
    "while",
    "as",
    "extends",
}


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.index = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while not self._is_at_end():
            ch = self._peek()
            if ch in " \t\r":
                self._advance()
                continue
            if ch == "\n":
                tokens.append(self._make_token("NEWLINE", "\n"))
                self._advance()
                continue
            if ch == "-" and self._peek_next() == "-":
                self._skip_comment()
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_name())
                continue
            if ch.isdigit():
                tokens.append(self._read_number())
                continue
            if ch == '"':
                for t in self._read_string():
                    tokens.append(t)
                continue
            tokens.append(self._read_symbol())

        tokens.append(Token("EOF", "", self.line, self.column))
        return tokens

    def _read_name(self) -> Token:
        line, col, start = self.line, self.column, self.index
        while not self._is_at_end() and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        value = self.source[start:self.index]
        kind = "KEYWORD" if value in KEYWORDS else "NAME"
        return Token(kind, value, line, col)

    def _read_number(self) -> Token:
        line, col, start = self.line, self.column, self.index
        saw_dot = False
        while not self._is_at_end():
            ch = self._peek()
            if ch.isdigit():
                self._advance()
                continue
            if ch == "." and not saw_dot and self._peek_next().isdigit():
                saw_dot = True
                self._advance()
                continue
            break
        return Token("NUMBER", self.source[start:self.index], line, col)

    def _read_string(self) -> List[Token]:
        line, col = self.line, self.column
        self._advance()
        result: List[Token] = []
        chars: List[str] = []
        while not self._is_at_end():
            ch = self._peek()
            if ch == '"':
                self._advance()
                if chars:
                    result.append(Token("STRING_PART", "".join(chars), line, col))
                if len(result) == 1 and result[0].kind == "STRING_PART":
                    return [Token("STRING", result[0].value, line, col)]
                return result
            if ch == "{":
                if chars:
                    result.append(Token("STRING_PART", "".join(chars), line, col))
                    chars = []
                result.append(Token("INTERP_OPEN", "{", self.line, self.column))
                self._advance()
                content = self._read_until_matching_brace()
                sub_tokens = Lexer(content).tokenize()[:-1]
                result.extend(sub_tokens)
                result.append(Token("INTERP_CLOSE", "}", self.line, self.column))
                self._advance()
                line, col = self.line, self.column
                continue
            if ch == "\\":
                self._advance()
                if self._is_at_end():
                    break
                esc = self._peek()
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                chars.append(mapping.get(esc, esc))
                self._advance()
                continue
            if ch == "\n":
                raise LexerError("Unterminated string literal", line, col)
            chars.append(ch)
            self._advance()
        raise LexerError("Unterminated string literal", line, col)

    def _read_until_matching_brace(self) -> str:
        start = self.index
        depth = 1
        while not self._is_at_end() and depth > 0:
            ch = self._peek()
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return self.source[start:self.index]
            self._advance()
        raise LexerError("Unmatched '{' in string interpolation", self.line, self.column)

    def _read_symbol(self) -> Token:
        line, col = self.line, self.column
        two = self._peek() + self._peek_next()
        two_char_tokens = {
            "|>": "PIPE",
            "=>": "ARROW",
            ">=": "GTE",
            "<=": "LTE",
            "==": "EQEQ",
            "!=": "NEQ",
            "..": "DOTDOT",
        }
        if two in two_char_tokens:
            self._advance()
            self._advance()
            return Token(two_char_tokens[two], two, line, col)

        one_char_tokens = {
            "(": "LPAREN",
            ")": "RPAREN",
            "{": "LBRACE",
            "}": "RBRACE",
            "[": "LBRACKET",
            "]": "RBRACKET",
            ",": "COMMA",
            ";": "SEMICOLON",
            ".": "DOT",
            ":": "COLON",
            "=": "EQ",
            "+": "PLUS",
            "-": "MINUS",
            "*": "STAR",
            "/": "SLASH",
            "%": "PERCENT",
            "<": "LT",
            ">": "GT",
        }
        ch = self._peek()
        if ch in one_char_tokens:
            self._advance()
            return Token(one_char_tokens[ch], ch, line, col)
        raise LexerError(f"Unexpected character '{ch}'", line, col)

    def _skip_comment(self) -> None:
        while not self._is_at_end() and self._peek() != "\n":
            self._advance()

    def _make_token(self, kind: str, value: str) -> Token:
        return Token(kind, value, self.line, self.column)

    def _peek(self) -> str:
        if self._is_at_end():
            return "\0"
        return self.source[self.index]

    def _peek_next(self) -> str:
        if self.index + 1 >= self.length:
            return "\0"
        return self.source[self.index + 1]

    def _advance(self) -> str:
        ch = self.source[self.index]
        self.index += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _is_at_end(self) -> bool:
        return self.index >= self.length


def tokenize(source: str) -> List[Token]:
    return Lexer(source).tokenize()

