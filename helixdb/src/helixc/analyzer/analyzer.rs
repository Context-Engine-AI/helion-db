//! Semantic analyzer for Helix‑QL.

use colored::Colorize;

use crate::helixc::parser::{helix_parser::*, location::Loc};

use std::collections::{HashMap, HashSet};

use super::{fix::Fix, pretty};

/// A single diagnostic to be surfaced to the editor.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: Loc,
    pub message: String,
    pub hint: Option<String>,
    pub filename: Option<String>,
    pub severity: DiagnosticSeverity,
    pub fix: Option<Fix>,
}

#[derive(Debug, Clone)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
    Empty,
}

impl Diagnostic {
    pub fn new(
        location: Loc,
        message: impl Into<String>,
        severity: DiagnosticSeverity,
        hint: Option<String>,
        fix: Option<Fix>,
    ) -> Self {
        Self {
            location,
            message: message.into(),
            hint,
            fix,
            filename: None,
            severity,
        }
    }

    pub fn render(&self, src: &str, filename: &str) -> String {
        pretty::render(self, src, filename)
    }
}

pub fn analyze(src: &Source) -> Vec<Diagnostic> {
    let mut ctx = Ctx::new(src);
    ctx.check_schema();
    ctx.check_queries();
    ctx.diagnostics
}

/// Internal working context shared by all passes.
struct Ctx<'a> {
    src: &'a Source,

    /// Quick look‑ups
    node_set: HashSet<&'a str>,
    vector_set: HashSet<&'a str>,
    edge_map: HashMap<&'a str, &'a EdgeSchema>,
    node_fields: HashMap<&'a str, HashMap<&'a str, &'a FieldType>>,
    edge_fields: HashMap<&'a str, HashMap<&'a str, &'a FieldType>>,
    vector_fields: HashMap<&'a str, HashMap<&'a str, &'a FieldType>>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Ctx<'a> {
    fn new(src: &'a Source) -> Self {
        // Build field look‑ups once
        let node_fields = src
            .node_schemas
            .iter()
            .map(|n| {
                (
                    n.name.1.as_str(),
                    n.fields
                        .iter()
                        .map(|f| (f.name.as_str(), &f.field_type))
                        .collect(),
                )
            })
            .collect();

        let edge_fields = src
            .edge_schemas
            .iter()
            .map(|e| {
                (
                    e.name.1.as_str(),
                    e.properties
                        .as_ref()
                        .map(|v| v.iter().map(|f| (f.name.as_str(), &f.field_type)).collect())
                        .unwrap_or_else(HashMap::new),
                )
            })
            .collect();

        let vector_fields = src
            .vector_schemas
            .iter()
            .map(|v| {
                (
                    v.name.as_str(),
                    v.fields
                        .iter()
                        .map(|f| (f.name.as_str(), &f.field_type))
                        .collect(),
                )
            })
            .collect();

        Self {
            node_set: src.node_schemas.iter().map(|n| n.name.1.as_str()).collect(),
            vector_set: src.vector_schemas.iter().map(|v| v.name.as_str()).collect(),
            edge_map: src
                .edge_schemas
                .iter()
                .map(|e| (e.name.1.as_str(), e))
                .collect(),
            node_fields,
            edge_fields,
            vector_fields,
            src,
            diagnostics: Vec::new(),
        }
    }

    // ---------- Pass #1: schema --------------------------
    /// Validate that every edge references declared node types.
    fn check_schema(&mut self) {
        for edge in &self.src.edge_schemas {
            if !self.node_set.contains(edge.from.1.as_str())
                && !self.vector_set.contains(edge.from.1.as_str())
            {
                self.push_schema_err(
                    &edge.from.1,
                    edge.from.0.clone(),
                    format!("`{}` is not a declared node type", edge.from.1),
                    Some(format!("Declare `N::{}` before this edge", edge.from.1)),
                );
            }
            if !self.node_set.contains(edge.to.1.as_str())
                && !self.vector_set.contains(edge.to.1.as_str())
            {
                self.push_schema_err(
                    &edge.to.1,
                    edge.to.0.clone(),
                    format!("`{}` is not a declared node type", edge.to.1),
                    Some(format!("Declare `N::{}` before this edge", edge.to.1)),
                );
            }
        }
    }

    // ---------- Pass #2: queries -------------------------
    fn check_queries(&mut self) {
        for q in &self.src.queries {
            self.check_query(q);
        }
    }

    fn check_query(&mut self, q: &'a Query) {
        // -------------------------------------------------
        // Parameter validation
        // -------------------------------------------------
        for param in &q.parameters {
            if let FieldType::Identifier(ref id) = param.param_type.1 {
                if !self.node_set.contains(id.as_str()) && !self.vector_set.contains(id.as_str()) {
                    self.push_query_err(
                        q,
                        param.param_type.0.clone(),
                        format!("unknown type `{}` for parameter `{}`", id, param.name.1),
                        "declare or use a matching schema object or use a primitive type",
                    );
                }
            }
        }

        // -------------------------------------------------
        // Statement‑by‑statement walk
        // -------------------------------------------------
        let mut scope: HashMap<&str, Type<'a>> = HashMap::new();
        for param in &q.parameters {
            scope.insert(param.name.1.as_str(), Type::from(&param.param_type.1));
        }

        use StatementType::*;
        for stmt in &q.statements {
            match &stmt.statement {
                Assignment(assign) => {
                    if scope.contains_key(assign.variable.as_str()) {
                        self.push_query_err(
                            q,
                            assign.loc.clone(),
                            format!("variable `{}` is already declared", assign.variable),
                            "rename the new variable or remove the previous definition",
                        );
                        continue;
                    }

                    let rhs_ty = self.infer_expr_type(&assign.value, &scope, q, None);
                    scope.insert(assign.variable.as_str(), rhs_ty);
                }

                AddNode(add) => {
                    if let Some(ref ty) = add.node_type {
                        if !self.node_set.contains(ty.as_str()) {
                            self.push_query_err(
                                q,
                                add.loc.clone(),
                                format!("`AddN<{}>` refers to unknown node type", ty),
                                "declare the node schema first",
                            );
                        }
                    }
                }

                AddEdge(add) => {
                    if let Some(ref ty) = add.edge_type {
                        if !self.edge_map.contains_key(ty.as_str()) {
                            self.push_query_err(
                                q,
                                add.loc.clone(),
                                format!("`AddE<{}>` refers to unknown edge type", ty),
                                "declare the edge schema first",
                            );
                        }
                    }
                }

                AddVector(add) => {
                    if let Some(ref ty) = add.vector_type {
                        if !self.vector_set.contains(ty.as_str()) {
                            self.push_query_err(
                                q,
                                add.loc.clone(),
                                format!("vector type `{}` has not been declared", ty),
                                "add a `V::{}` schema first",
                            );
                        }
                    }
                }

                BatchAddVector(add) => {
                    if let Some(ref ty) = add.vector_type {
                        if !self.vector_set.contains(ty.as_str()) {
                            self.push_query_err(
                                q,
                                add.loc.clone(),
                                format!("vector type `{}` has not been declared", ty),
                                "add a `V::{}` schema first",
                            );
                        }
                    }
                }

                Drop(expr) => {
                    // Nothing special right now; still type‑check sub‑expressions
                    self.infer_expr_type(expr, &scope, q, None);
                }

                SearchVector(expr) => {
                    if let Some(ref ty) = expr.vector_type {
                        if !self.vector_set.contains(ty.as_str()) {
                            self.push_query_err(
                                q,
                                expr.loc.clone(),
                                format!("vector type `{}` has not been declared", ty),
                                "add a `V::{}` schema first",
                            );
                        }
                    }
                }

                ForLoop(fl) => {
                    // Ensure the collection exists
                    if !scope.contains_key(fl.in_variable.as_str()) {
                        self.push_query_err(
                            q,
                            fl.loc.clone(),
                            format!("`{}` is not defined in the current scope", fl.in_variable),
                            "add a statement assigning it before the loop",
                        );
                    }
                    // Add loop vars to new child scope and walk the body
                    let mut body_scope = scope.clone();
                    for v in &fl.variables {
                        body_scope.insert(v.as_str(), Type::Unknown);
                    }
                    for body_stmt in &fl.statements {
                        // Recursive walk (but without infinite nesting for now)
                        if let StatementType::Assignment(a) = &body_stmt.statement {
                            let t = self.infer_expr_type(&a.value, &body_scope, q, None);
                            body_scope.insert(a.variable.as_str(), t);
                        }
                    }
                }
            }
        }

        // -------------------------------------------------
        // Validate RETURN expressions
        // -------------------------------------------------
        if q.return_values.is_empty() {
            let end = q.loc.end.clone();
            self.push_query_warn(
                q,
                Loc::new(end.clone(), end, q.loc.span.clone()),
                "query has no RETURN clause".to_string(),
                "add `RETURN <expr>` at the end",
                None,
            );
        }
        for ret in &q.return_values {
            self.infer_expr_type(ret, &scope, q, None);
        }
    }

    // -----------------------------------------------------
    // Helpers
    // -----------------------------------------------------
    fn push_schema_err(&mut self, _ident: &str, loc: Loc, msg: String, hint: Option<String>) {
        self.diagnostics.push(Diagnostic::new(
            loc,
            msg,
            DiagnosticSeverity::Error,
            hint,
            None,
        ));
    }
    fn push_query_err(&mut self, q: &Query, loc: Loc, msg: String, hint: impl Into<String>) {
        self.diagnostics.push(Diagnostic::new(
            loc,
            format!("{} (in QUERY named `{}`)", msg, q.name),
            DiagnosticSeverity::Error,
            Some(hint.into()),
            None,
        ));
    }

    fn push_query_err_with_fix(
        &mut self,
        q: &Query,
        loc: Loc,
        msg: String,
        hint: impl Into<String>,
        fix: Fix,
    ) {
        self.diagnostics.push(Diagnostic::new(
            loc,
            format!("{} (in QUERY named `{}`)", msg, q.name),
            DiagnosticSeverity::Error,
            Some(hint.into()),
            Some(fix),
        ));
    }

    fn push_query_warn(
        &mut self,
        q: &Query,
        loc: Loc,
        msg: String,
        hint: impl Into<String>,
        fix: Option<Fix>,
    ) {
        self.diagnostics.push(Diagnostic::new(
            loc,
            format!("{} (in QUERY named `{}`)", msg, q.name),
            DiagnosticSeverity::Warning,
            Some(hint.into()),
            fix,
        ));
    }

    /// Infer the semantic `Type` of an expression *and* perform all traversal
    ///‑specific validations (property access, field exclusions, step ordering).
    fn infer_expr_type(
        &mut self,
        expression: &'a Expression,
        scope: &HashMap<&'a str, Type<'a>>,
        q: &'a Query,
        parent_ty: Option<Type<'a>>,
    ) -> Type<'a> {
        use ExpressionType::*;
        let expr = &expression.expr;
        match expr {
            Identifier(name) => match scope.get(name.as_str()) {
                Some(t) => t.clone(),
                None => {
                    self.push_query_err(
                        q,
                        expression.loc.clone(),
                        format!("variable named `{}` is not in scope", name),
                        "declare it earlier or fix the typo",
                    );
                    Type::Unknown
                }
            },

            IntegerLiteral(_) => Type::Scalar(FieldType::I32),
            FloatLiteral(_) => Type::Scalar(FieldType::F64),
            StringLiteral(_) => Type::Scalar(FieldType::String),
            BooleanLiteral(_) => Type::Boolean,
            Empty => Type::Unknown,

            Traversal(tr) | Exists(tr) => {
                let final_ty = self.check_traversal(tr, scope, q, parent_ty);
                if matches!(expr, Exists(_)) {
                    Type::Boolean
                } else {
                    final_ty
                }
            }

            AddNode(add) => {
                if let Some(ref ty) = add.node_type {
                    if !self.node_set.contains(ty.as_str()) {
                        self.push_query_err(
                            q,
                            add.loc.clone(),
                            format!("`AddN<{}>` refers to unknown node type", ty),
                            "declare the node schema first",
                        );
                    }
                    // Validate fields if both type and fields are present
                    if let Some(fields) = &add.fields {
                        // Get the field set before validation
                        let field_set = self.node_fields.get(ty.as_str()).cloned();
                        if let Some(field_set) = field_set {
                            for (field_name, _) in fields {
                                if !field_set.contains_key(field_name.as_str()) {
                                    self.push_query_err(
                                        q,
                                        add.loc.clone(),
                                        format!("`{}` is not a field of node `{}`", field_name, ty),
                                        "check the schema field names",
                                    );
                                }
                            }
                        }
                    }
                }
                Type::Nodes(add.node_type.as_deref())
            }
            AddEdge(add) => {
                if let Some(ref ty) = add.edge_type {
                    if !self.edge_map.contains_key(ty.as_str()) {
                        self.push_query_err(
                            q,
                            add.loc.clone(),
                            format!("`AddE<{}>` refers to unknown edge type", ty),
                            "declare the edge schema first",
                        );
                    }
                    // Validate fields if both type and fields are present
                    if let Some(fields) = &add.fields {
                        // Get the field set before validation
                        let field_set = self.edge_fields.get(ty.as_str()).cloned();
                        if let Some(field_set) = field_set {
                            for (field_name, _) in fields {
                                if !field_set.contains_key(field_name.as_str()) {
                                    self.push_query_err(
                                        q,
                                        add.loc.clone(),
                                        format!("`{}` is not a field of edge `{}`", field_name, ty),
                                        "check the schema field names",
                                    );
                                }
                            }
                        }
                    }
                }
                Type::Edges(add.edge_type.as_deref())
            }
            AddVector(add) => {
                if let Some(ref ty) = add.vector_type {
                    if !self.vector_set.contains(ty.as_str()) {
                        self.push_query_err(
                            q,
                            add.loc.clone(),
                            format!("vector type `{}` has not been declared", ty),
                            format!("add a `V::{}` schema first", ty),
                        );
                    }
                    // Validate vector fields
                    if let Some(fields) = &add.fields {
                        let field_set = self.vector_fields.get(ty.as_str()).cloned();
                        if let Some(field_set) = field_set {
                            for (field_name, _) in fields {
                                if !field_set.contains_key(field_name.as_str()) {
                                    self.push_query_err(
                                        q,
                                        add.loc.clone(),
                                        format!(
                                            "`{}` is not a field of vector `{}`",
                                            field_name, ty
                                        ),
                                        "check the schema field names",
                                    );
                                }
                            }
                        }
                    }
                }
                Type::Vector(add.vector_type.as_deref())
            }
            BatchAddVector(add) => {
                if let Some(ref ty) = add.vector_type {
                    if !self.vector_set.contains(ty.as_str()) {
                        self.push_query_err(
                            q,
                            add.loc.clone(),
                            format!("vector type `{}` has not been declared", ty),
                            format!("add a `V::{}` schema first", ty),
                        );
                    }
                }
                Type::Vector(add.vector_type.as_deref())
            }
            SearchVector(sv) => {
                if let Some(ref ty) = sv.vector_type {
                    if !self.vector_set.contains(ty.as_str()) {
                        self.push_query_err(
                            q,
                            sv.loc.clone(),
                            format!("vector type `{}` has not been declared", ty),
                            format!("add a `V::{}` schema first", ty),
                        );
                    }
                }
                // Search returns nodes that contain the vectors
                Type::Nodes(None)
            }
            And(v) | Or(v) => {
                for e in v {
                    self.infer_expr_type(e, scope, q, parent_ty.clone());
                }
                Type::Boolean
            }
        }
    }

    // -----------------------------------------------------
    // Traversal‑specific checks
    // -----------------------------------------------------
    fn check_traversal(
        &mut self,
        tr: &'a Traversal,
        scope: &HashMap<&'a str, Type<'a>>,
        q: &'a Query,
        parent_ty: Option<Type<'a>>,
    ) -> Type<'a> {
        let mut previous_step = None;
        let mut cur_ty = match &tr.start {
            StartNode::Node { types, .. } => {
                if let Some(ref ts) = types {
                    // Check node types exist
                    for t in ts {
                        if !self.node_set.contains(t.as_str()) {
                            self.push_query_err(
                                q,
                                tr.loc.clone(),
                                format!("unknown node type `{}`", t),
                                format!("declare N::{} in the schema first", t),
                            );
                        }
                    }
                    Type::Nodes(ts.first().map(|s| s.as_str()))
                } else {
                    Type::Nodes(None)
                }
            }
            StartNode::Edge { types, .. } => {
                if let Some(ref ts) = types {
                    for t in ts {
                        if !self.edge_map.contains_key(t.as_str()) {
                            self.push_query_err(
                                q,
                                tr.loc.clone(),
                                format!("unknown edge type `{}`", t),
                                format!("declare E::{} in the schema first", t),
                            );
                        }
                    }
                    Type::Edges(ts.first().map(|s| s.as_str()))
                } else {
                    Type::Edges(None)
                }
            }
            StartNode::Variable(var) => scope.get(var.as_str()).cloned().unwrap_or_else(|| {
                self.push_query_err(
                    q,
                    tr.loc.clone(),
                    format!("variable named `{}` is not in scope", var),
                    format!("declare {} in the current scope or fix the typo", var),
                );
                Type::Unknown
            }),
            StartNode::Anonymous => parent_ty.unwrap_or(Type::Unknown),
        };

        // Track excluded fields for property validation
        let mut excluded: HashMap<&str, Loc> = HashMap::new();

        // Stream through the steps
        for graph_step in &tr.steps {
            let step = &graph_step.step;
            match step {
                StepType::Node(gs) | StepType::Edge(gs) => {
                    match self.apply_graph_step(&gs, &cur_ty, q) {
                        Some(new_ty) => cur_ty = new_ty,
                        None => { /* error already recorded */ }
                    }
                    excluded.clear(); // Traversal to a new element resets exclusions
                }

                StepType::Count => {
                    cur_ty = Type::Scalar(FieldType::I64);
                    excluded.clear();
                }

                StepType::Exclude(ex) => {
                    self.validate_exclude(&cur_ty, tr, ex, &excluded, q);
                    for (_, key) in &ex.fields {
                        excluded.insert(key.as_str(), ex.loc.clone());
                    }
                }

                StepType::Object(obj) => {
                    self.validate_object(&cur_ty, tr, obj, &excluded, q);
                }

                StepType::Where(expr) => {
                    self.infer_expr_type(expr, scope, q, Some(cur_ty.clone()));
                    // Where/boolean ops don't change the element type,
                    // so `cur_ty` stays the same.
                }

                StepType::BooleanOperation(b_op) => {
                    let step = previous_step.unwrap();
                    let property_type = match &b_op.op {
                        BooleanOpType::LessThanOrEqual(expr)
                        | BooleanOpType::LessThan(expr)
                        | BooleanOpType::GreaterThanOrEqual(expr)
                        | BooleanOpType::GreaterThan(expr)
                        | BooleanOpType::Equal(expr)
                        | BooleanOpType::NotEqual(expr) => {
                            match self.infer_expr_type(expr, scope, q, Some(cur_ty.clone())) {
                                Type::Scalar(ft) => ft.clone(),
                                field_type => {
                                    self.push_query_err(
                                        q,
                                        b_op.loc.clone(),
                                        format!("boolean operation `{}` cannot be applied to `{}`", b_op.loc.span, field_type.kind_str()),
                                        "make sure the expression evaluates to a number or a string".to_string(),
                                    );
                                    return field_type;
                                }
                            }
                        }
                        _ => return cur_ty.clone(),
                    };

                    // get type of field name
                    let field_name = match step {
                        StepType::Object(obj) => {
                            let fields = obj.fields;
                            assert!(fields.len() == 1);
                            Some(fields[0].1.value.clone())
                        }
                        _ => None,
                    };
                    if let Some(FieldValueType::Identifier(field_name)) = &field_name {
                        match &cur_ty {
                            Type::Nodes(Some(node_ty)) => {
                                let field_set = self.node_fields.get(node_ty).cloned();
                                if let Some(field_set) = field_set {
                                    match field_set.get(field_name.as_str()) {
                                        Some(field) => {
                                            if field != &&property_type {
                                                self.push_query_err(
                                                    q,
                                                    b_op.loc.clone(),
                                                    format!("property `{field_name}` is of type `{field}` (from node type `{node_ty}::{{{field_name}}}`), which does not match type of compared value `{property_type}`"),
                                                    "make sure comparison value is of the same type as the property".to_string(),
                                                );
                                            }
                                        }
                                        None => {
                                            self.push_query_err(
                                                q,
                                                b_op.loc.clone(),
                                                format!(
                                                    "`{}` is not a field of {} `{}`",
                                                    field_name, "node", node_ty
                                                ),
                                                "check the schema field names",
                                            );
                                        }
                                    }
                                }
                            }
                            Type::Edges(Some(edge_ty)) => {
                                let field_set = self.edge_fields.get(edge_ty).cloned();
                                if let Some(field_set) = field_set {
                                    match field_set.get(field_name.as_str()) {
                                        Some(field) => {
                                            if field != &&property_type {
                                                self.push_query_err(
                                                    q,
                                                    b_op.loc.clone(),
                                                    format!("property `{field_name}` is of type `{field}` (from edge type `{edge_ty}::{{{field_name}}}`), which does not match type of compared value `{property_type}`"),
                                                    "make sure comparison value is of the same type as the property".to_string(),
                                                );
                                            }
                                        }
                                        None => {
                                            self.push_query_err(
                                                q,
                                                b_op.loc.clone(),
                                                format!(
                                                    "`{}` is not a field of {} `{}`",
                                                    field_name, "edge", edge_ty
                                                ),
                                                "check the schema field names",
                                            );
                                        }
                                    }
                                }
                            }
                            _ => {
                                self.push_query_err(
                                    q,
                                    b_op.loc.clone(),
                                    "boolean operation can only be applied to scalar values"
                                        .to_string(),
                                    "make sure the expression evaluates to a number or a string"
                                        .to_string(),
                                );
                            }
                        }
                    }

                    // self.infer_expr_type(expr, scope, q);
                    // Where/boolean ops don't change the element type,
                    // so `cur_ty` stays the same.
                }

                StepType::Update(_) => {
                    // Update returns the same type (nodes/edges) it started with.
                    excluded.clear();
                }

                StepType::AddEdge(add) => {
                    if let Some(ref ty) = add.edge_type {
                        if !self.edge_map.contains_key(ty.as_str()) {
                            self.push_query_err(
                                q,
                                add.loc.clone(),
                                format!("`AddE<{}>` refers to unknown edge type", ty),
                                "declare the edge schema first",
                            );
                        }
                    }
                    cur_ty = Type::Edges(add.edge_type.as_deref());
                    excluded.clear();
                }

                StepType::Range(_) => { /* doesn't affect type */ }
                StepType::Closure(_cl) => {
                    // Add identifier to a temporary scope so inner uses pass
                    // let mut tmp_scope = scope.clone();
                    // tmp_scope.insert(cl.identifier.as_str(), cur_ty.clone());
                    // // Walk the object literal
                    // let obj_expr = Expression::Object(Object {
                    //     fields: cl.object.fields.clone(),
                    //     should_spread: cl.object.should_spread,
                    // });
                    // self.infer_expr_type(&obj_expr, &tmp_scope, q);
                    // cur_ty = Type::Unknown;
                }

                StepType::SearchVector(_) => {
                    // SearchV on a traversal returns nodes again
                    cur_ty = Type::Nodes(None);
                    excluded.clear();
                }
            }
            previous_step = Some(step.clone());
        }

        cur_ty
    }

    fn validate_exclude_fields(
        &mut self,
        ex: &Exclude,
        field_set: &HashMap<&str, &FieldType>,
        excluded: &HashMap<&str, Loc>,
        q: &'a Query,
        type_name: &str,
        type_kind: &str,
        span: Option<Loc>,
    ) {
        for (loc, key) in &ex.fields {
            if let Some(loc) = excluded.get(key.as_str()) {
                self.push_query_err_with_fix(
                    q,
                    loc.clone(),
                    format!("field `{}` was previously excluded in this traversal", key),
                    format!("remove the exclusion of `{}`", key),
                    Fix::new(span.clone(), Some(loc.clone()), None),
                );
            } else if !field_set.contains_key(key.as_str()) {
                self.push_query_err(
                    q,
                    loc.clone(),
                    format!("`{}` is not a field of {} `{}`", key, type_kind, type_name),
                    "check the schema field names",
                );
            }
        }
    }

    fn validate_exclude(
        &mut self,
        cur_ty: &Type<'a>,
        tr: &'a Traversal,
        ex: &Exclude,
        excluded: &HashMap<&str, Loc>,
        q: &'a Query,
    ) {
        match &cur_ty {
            Type::Nodes(Some(node_ty)) => {
                if let Some(field_set) = self.node_fields.get(node_ty).cloned() {
                    self.validate_exclude_fields(
                        ex,
                        &field_set,
                        &excluded,
                        q,
                        node_ty,
                        "node",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Edges(Some(edge_ty)) => {
                // for (key, val) in &obj.fields {
                if let Some(field_set) = self.edge_fields.get(edge_ty).cloned() {
                    self.validate_exclude_fields(
                        ex,
                        &field_set,
                        &excluded,
                        q,
                        edge_ty,
                        "edge",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Vector(Some(vector_ty)) => {
                // Vectors only have 'id' and 'embedding' fields
                if let Some(fields) = self.vector_fields.get(vector_ty).cloned() {
                    self.validate_exclude_fields(
                        ex,
                        &fields,
                        &excluded,
                        q,
                        vector_ty,
                        "vector",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Anonymous(ty) => {
                self.validate_exclude(ty, tr, ex, excluded, q);
            }
            Type::Nodes(None) | Type::Edges(None) | Type::Vector(None) => {}
            _ => {
                self.push_query_err(
                    q,
                    ex.fields[0].0.clone(),
                    "cannot access properties on this type".to_string(),
                    "exclude is only valid on nodes, edges and vectors",
                );
            }
        }
    }

    fn validate_object(
        &mut self,
        cur_ty: &Type<'a>,
        tr: &'a Traversal,
        obj: &Object,
        excluded: &HashMap<&str, Loc>,
        q: &'a Query,
    ) {
        match &cur_ty {
            Type::Nodes(Some(node_ty)) => {
                if let Some(field_set) = self.node_fields.get(node_ty).cloned() {
                    self.validate_object_fields(
                        obj,
                        &field_set,
                        &excluded,
                        q,
                        node_ty,
                        "node",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Edges(Some(edge_ty)) => {
                // for (key, val) in &obj.fields {
                if let Some(field_set) = self.edge_fields.get(edge_ty).cloned() {
                    self.validate_object_fields(
                        obj,
                        &field_set,
                        &excluded,
                        q,
                        edge_ty,
                        "edge",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Vector(Some(vector_ty)) => {
                // Vectors only have 'id' and 'embedding' fields
                if let Some(fields) = self.vector_fields.get(vector_ty).cloned() {
                    self.validate_object_fields(
                        obj,
                        &fields,
                        &excluded,
                        q,
                        vector_ty,
                        "vector",
                        Some(tr.loc.clone()),
                    );
                }
            }
            Type::Anonymous(ty) => {
                self.validate_object(ty, tr, obj, excluded, q);
            }
            Type::Nodes(None) | Type::Edges(None) | Type::Vector(None) => {}
            _ => {
                self.push_query_err(
                    q,
                    obj.fields[0].1.loc.clone(),
                    "cannot access properties on this type".to_string(),
                    "property access is only valid on nodes, edges and vectors",
                );
            }
        }
    }

    /// Check that a graph‑navigation step is allowed for the current element
    /// kind and return the post‑step kind.
    fn apply_graph_step(
        &mut self,
        gs: &'a GraphStep,
        cur_ty: &Type<'a>,
        q: &'a Query,
    ) -> Option<Type<'a>> {
        use GraphStepType::*;
        match (&gs.step, cur_ty.base()) {
            // Node‑to‑Edge
            (OutE(_) | InE(_), Type::Nodes(_)) => Some(Type::Edges(Some(
                gs.loc
                    .span
                    .trim_matches(|c: char| c == '"' || c.is_whitespace() || c == '\n')
                    .trim_start_matches("OutE<")
                    .trim_start_matches("InE<")
                    .trim_end_matches(">"),
            ))),

            // Node‑to‑Node
            (Out(_) | In(_), Type::Nodes(_)) => Some(Type::Nodes(Some(
                gs.loc
                    .span
                    .trim_matches(|c: char| c == '"' || c.is_whitespace() || c == '\n')
                    .trim_start_matches("Out<")
                    .trim_start_matches("In<")
                    .trim_end_matches(">"),
            ))),

            // Edge‑to‑Node
            (FromN | ToN, Type::Edges(_)) => {
                Some(Type::Nodes(Some(gs.loc.span.trim_matches(|c: char| {
                    c == '"' || c.is_whitespace() || c == '\n'
                }))))
            }

            // Anything else is illegal
            _ => {
                self.push_query_err(
                    q,
                    gs.loc.clone(),
                    format!(
                        "traversal step `{}` cannot follow a step that returns {}",
                        gs.loc
                            .span
                            .trim_matches(|c: char| c == '"' || c.is_whitespace() || c == '\n')
                            .bold(),
                        cur_ty
                            .kind_str()
                            .trim_matches(|c: char| c == '"' || c.is_whitespace() || c == '\n')
                            .bold()
                    ),
                    self.get_traversal_step_hint(cur_ty, &gs.step).as_str(),
                );
                None
            }
        }
    }

    fn get_traversal_step_hint(
        &self,
        current_step: &Type<'a>,
        next_step: &GraphStepType,
    ) -> String {
        match (current_step, next_step) {
            (Type::Nodes(Some(span)), GraphStepType::ToN | GraphStepType::FromN) => {
                format!(
                    "\n{}\n{}",
                    format!(
                        "      • Use `OutE` or `InE` to traverse edges from `{}`",
                        span
                    ),
                    format!(
                        "      • Use `Out` or `In` to traverse nodes from `{}`",
                        span
                    ),
                )
            }
            (Type::Edges(Some(span)), GraphStepType::OutE(_) | GraphStepType::InE(_)) => {
                format!("use `FromN` or `ToN` to traverse nodes from `{}`", span)
            }
            (Type::Edges(Some(span)), GraphStepType::Out(_) | GraphStepType::In(_)) => {
                format!("use `FromN` or `ToN` to traverse nodes from `{}`", span)
            }
            (_, _) => "re-order the traversal or remove the invalid step".to_string(),
        }
    }

    fn validate_object_fields(
        &mut self,
        obj: &Object,
        field_set: &HashMap<&str, &FieldType>,
        excluded: &HashMap<&str, Loc>,
        q: &'a Query,
        type_name: &str,
        type_kind: &str,
        span: Option<Loc>,
    ) {
        for (key, val) in &obj.fields {
            if let Some(loc) = excluded.get(key.as_str()) {
                // for the "::"
                let mut loc = loc.clone();
                loc.end.column += 2;
                loc.span.push_str("::");
                self.push_query_err_with_fix(
                    q,
                    val.loc.clone(),
                    format!("field `{}` was previously excluded in this traversal", key),
                    format!("remove the exclusion of `{}`", key),
                    Fix::new(span.clone(), Some(loc.clone()), Some(String::new())),
                );
            } else if !field_set.contains_key(key.as_str()) {
                self.push_query_err(
                    q,
                    val.loc.clone(),
                    format!("`{}` is not a field of {} `{}`", key, type_kind, type_name),
                    "check the schema field names",
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Type<'a> {
    Nodes(Option<&'a str>),
    Edges(Option<&'a str>),
    Vector(Option<&'a str>),
    Scalar(FieldType),
    #[allow(dead_code)]
    Anonymous(Box<Type<'a>>),
    Boolean,
    Unknown,
}

impl<'a> Type<'a> {
    fn kind_str(&self) -> &'static str {
        match self {
            Type::Nodes(_) => "nodes",
            Type::Edges(_) => "edges",
            Type::Vector(_) => "vectors",
            Type::Scalar(_) => "scalar",
            Type::Boolean => "boolean",
            Type::Unknown => "unknown",
            Type::Anonymous(ty) => ty.kind_str(),
        }
    }

    /// Recursively strip <code>Anonymous</code> layers and return the base type.
    fn base(&self) -> &Type<'a> {
        match self {
            Type::Anonymous(inner) => inner.base(),
            _ => self,
        }
    }

    /// Same, but returns an owned clone for convenience.
    #[allow(dead_code)]
    fn cloned_base(&self) -> Type<'a> {
        match self {
            Type::Anonymous(inner) => inner.cloned_base(),
            _ => self.clone(),
        }
    }
}

impl<'a> From<&'a FieldType> for Type<'a> {
    fn from(ft: &'a FieldType) -> Self {
        use FieldType::*;
        match ft {
            String | Boolean | F32 | F64 | I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | U128 => {
                Type::Scalar(ft.clone())
            }
            Array(_) | Object(_) | Identifier(_) => Type::Unknown,
        }
    }
}

// ---------------------------------
// Tests
// ---------------------------------
#[cfg(test)]
mod analyzer_tests {
    use super::*;
    use crate::helixc::parser::helix_parser::HelixParser;

    /// Convenience helper – parse text and return diagnostics.
    fn run(src: &str) -> Vec<Diagnostic> {
        let input = write_to_temp_file(vec![src]);
        let parsed = HelixParser::parse_source(&input)
            .expect("parser should succeed – these tests are for the analyzer");
        analyze(&parsed)
    }

    #[test]
    fn reports_unknown_node_in_edge() {
        let hx = r#"
            E::Likes {
                From: User,
                To: Post,
                Properties: {}
            }
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("not a declared node type")),
            "expected a diagnostic about undeclared node types, got: {:?}",
            diags
        );
    }

    #[test]
    fn detects_redeclared_variable() {
        let hx = r#"
            N::User { Name: String }

            QUERY dupVar() =>
                u <- N<User>
                u <- N<User>
                RETURN u
        "#;
        let diags = run(hx);
        assert!(
            diags.iter().any(|d| d.message.contains("already declared")),
            "expected a diagnostic about variable redeclaration, got: {:?}",
            diags
        );
    }

    #[test]
    fn flags_invalid_property_access() {
        let hx = r#"
            N::User { name: String }

            QUERY badField() =>
                u <- N<User>
                n <- u::{age}
                RETURN n
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of node")),
            "expected a diagnostic about invalid property access, got: {:?}",
            diags
        );
    }

    #[test]
    fn traversal_step_order_enforced() {
        let hx = r#"
            N::User { name: String }

            QUERY wrongStep() =>
                e <- N<User>::FromN   // OutN on nodes is illegal (needs Edge)
                RETURN e
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("cannot follow a step that returns")),
            "expected a diagnostic about illegal step ordering, got: {:?}",
            diags
        );
    }

    #[test]
    fn clean_query_produces_no_diagnostics() {
        let hx = r#"
            N::User { name: String }
            N::Post { title: String }
            E::Wrote {
                From: User,
                To: Post,
                Properties: {}
            }

            QUERY ok(hey: User) =>
                u <- N<User>
                p <- u::Out<Wrote>
                RETURN p::!{title}::{title}
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags.is_empty(),
            "expected no diagnostics, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_edge_properties() {
        let hx = r#"
            N::User { name: String }
            N::Post { title: String }
            E::Wrote {
                From: User,
                To: Post,
                Properties: {
                    date: String,
                    likes: I32
                }
            }

            QUERY badEdgeField() =>
                e <- N<User>::OutE<Wrote>
                n <- e::{invalid_field}
                RETURN n
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of edge")),
            "expected a diagnostic about invalid edge property access, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_node_properties() {
        let hx = r#"
            N::User { name: String }
            N::Post { title: String }

            QUERY badNodeField() =>
                n <- N<User>::{invalid_field}
                RETURN n
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of node")),
            "expected a diagnostic about invalid node property access, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_vector_properties() {
        let hx = r#"
            V::UserEmbedding {
                content: String
            }

            QUERY badVectorField(vec: [F64], content: String) =>
                v <- AddV<UserEmbedding>(vec,{content: content})
                bad <- v::{invalid_field}
                RETURN bad
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of vector")),
            "expected a diagnostic about invalid vector property access, got: {:?}",
            diags
        );
    }

    #[test]
    fn handles_untyped_nodes() {
        let hx = r#"
            N::User { name: String }

            QUERY untypedNode() =>
                u <- N::{some_field}
                RETURN u
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags.is_empty(),
            "expected no diagnostics for untyped node access, got: {:?}",
            diags
        );
    }

    #[test]
    fn respects_excluded_fields() {
        let hx = r#"
            N::User { name: String, age: I32 }

            QUERY excludedField() =>
                u <- N<User>
                n <- u::!{name}::{name}
                RETURN n
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("was previously excluded")),
            "expected a diagnostic about accessing excluded field, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_add_node_fields() {
        let hx = r#"
            N::User { name: String }

            QUERY badAddNodeField() =>
                n <- AddN<User>({invalid_field: "test"})
                RETURN n
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of node")),
            "expected a diagnostic about invalid node field, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_add_edge_fields() {
        let hx = r#"
            N::User { name: String }
            N::Post { title: String }
            E::Wrote {
                From: User,
                To: Post,
                Properties: {
                    date: String
                }
            }

            QUERY badAddEdgeField() =>
                n1 <- AddN<User>({name: "test"})
                n2 <- AddN<Post>({title: "test"})
                e <- AddE<Wrote>({invalid_field: "test"})::To(n1)::From(n2)
                RETURN e
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of edge")),
            "expected a diagnostic about invalid edge field, got: {:?}",
            diags
        );
    }

    #[test]
    fn validates_add_vector_fields() {
        let hx = r#"
            V::UserEmbedding {
                content: String
            }

            QUERY badAddVectorField() =>
                v <- AddV<UserEmbedding>([1.0, 2.0], {invalid_field: "test"})
                RETURN v
        "#;
        let diags = run(hx);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("is not a field of vector")),
            "expected a diagnostic about invalid vector field, got: {:?}",
            diags
        );
    }

    #[test]
    fn validate_boolean_comparison() {
        let hx = r#"
            N::User { name: String }

            QUERY booleanComparison() =>
                a <- N<User>::WHERE(_::{name}::EQ(10))
                RETURN a
        "#;
        let diags = run(hx);
        for d in diags.iter() {
            println!("{}", d.render(hx, "query.hx"));
        }
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("does not match type of compared value")),
            "expected a type-mismatch diagnostic, got: {:?}",
            diags
        );
    }
}
