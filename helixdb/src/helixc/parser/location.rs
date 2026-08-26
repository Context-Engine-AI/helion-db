use pest::{iterators::Pair, Position};

use super::helix_parser::Rule;

#[derive(Debug, Clone)]
pub struct Loc {
    // pub filename: String,
    pub start: Span,
    pub end: Span,
    pub span: String,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column: column + 1,
        }
    }

    pub fn from_pos(pos: &Position) -> Self {
        let (line, column) = pos.line_col();
        Self {
            line,
            column: column + 1,
        }
    }
}

impl Loc {
    pub fn new(start: Span, end: Span, span: String) -> Self {
        Self {
            // filename,
            start,
            end,
            span,
        }
    }

    pub fn empty() -> Self {
        Self::new(Span::new(1, 1), Span::new(1, 1), "".to_string())
    }
}

pub trait HasLoc {
    fn loc(&self) -> Loc;
}

impl<'a> HasLoc for Pair<'a, Rule> {
    fn loc(&self) -> Loc {
        Loc::new(
            Span::from_pos(&self.as_span().start_pos()),
            Span::from_pos(&self.as_span().end_pos()),
            self.as_span().as_str().to_string(),
        )
    }
}
