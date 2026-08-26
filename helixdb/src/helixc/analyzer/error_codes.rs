pub enum ErrorCode {
    // Language errors
    UndeclaredType,
    VariableNotInScope,
    VariablePreviouslyDeclared,

    // Schema Type Errors
    InaccessibleProperty,
    FieldPreviouslyExcluded,
    FieldNotInSchema,
    EdgeNodeTypeNotInSchema,

    // Traversal errors
    WrongGraphStepOrder,
}

pub enum ErrorCode {
    E101,
    E102,
    E103,
    E104,
    E105,
    E106,
}