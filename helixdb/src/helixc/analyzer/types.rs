use std::collections::HashMap;

use crate::{
    helixc::parser::helix_parser::{
        AddEdge as ParserAddEdge, AddNode as ParserAddNode, AddVector as ParserAddVector,
        BatchAddVector as ParserBatchAddVector, BooleanOp as ParserBooleanOp,
        BooleanOpType as ParserBooleanOpType, Closure as ParserClosure,
        EdgeConnection as ParserEdgeConnection, EdgeSchema as ParserEdgeSchema,
        EvaluatesToNumber as ParserEvaluatesToNumber,
        EvaluatesToNumberType as ParserEvaluatesToNumberType, Exclude as ParserExclude,
        Expression as ParserExpression, ExpressionType as ParserExpressionType,
        Field as ParserField, FieldAddition as ParserFieldAddition, FieldType,
        FieldValue as ParserFieldValue, FieldValueType as ParserFieldValueType,
        GraphStep as ParserGraphStep, GraphStepType as ParserGraphStepType, IdType as ParserIdType,
        NodeSchema as ParserNodeSchema, Object as ParserObject, Query as ParserQuery,
        SearchVector as ParserSearchVector, Source as ParserSource, StartNode as ParserStartNode,
        Statement as ParserStatement, StatementType as ParserStatementType, Step as ParserStep,
        StepType as ParserStepType, Traversal as ParserTraversal, Update as ParserUpdate,
        ValueType as ParserValueType, VectorData as ParserVectorData,
        VectorSchema as ParserVectorSchema,
    },
    protocol::value::Value,
};

#[derive(Debug, Clone)]
pub struct Source {
    pub node_schemas: Vec<NodeSchema>,
    pub edge_schemas: Vec<EdgeSchema>,
    pub vector_schemas: Vec<VectorSchema>,
    pub queries: Vec<Query>,
}

impl From<ParserSource> for Source {
    fn from(source: ParserSource) -> Self {
        Source {
            node_schemas: source
                .node_schemas
                .into_iter()
                .map(NodeSchema::from)
                .collect(),
            edge_schemas: source
                .edge_schemas
                .into_iter()
                .map(EdgeSchema::from)
                .collect(),
            vector_schemas: source
                .vector_schemas
                .into_iter()
                .map(VectorSchema::from)
                .collect(),
            queries: source.queries.into_iter().map(Query::from).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeSchema {
    pub name: String,
    pub fields: Vec<Field>,
}

impl From<ParserNodeSchema> for NodeSchema {
    fn from(node_schema: ParserNodeSchema) -> Self {
        NodeSchema {
            name: node_schema.name.1,
            fields: node_schema
                .fields
                .into_iter()
                .map(|f| Field {
                    name: f.name,
                    field_type: f.field_type,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorSchema {
    pub name: String,
    pub fields: Vec<Field>,
}

impl From<ParserVectorSchema> for VectorSchema {
    fn from(vector_schema: ParserVectorSchema) -> Self {
        VectorSchema {
            name: vector_schema.name,
            fields: vector_schema
                .fields
                .into_iter()
                .map(|f| Field {
                    name: f.name,
                    field_type: f.field_type,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdgeSchema {
    pub name: String,
    pub from: String,
    pub to: String,
    pub properties: Option<Vec<Field>>,
}

impl From<ParserEdgeSchema> for EdgeSchema {
    fn from(edge_schema: ParserEdgeSchema) -> Self {
        EdgeSchema {
            name: edge_schema.name.1,
            from: edge_schema.from.1,
            to: edge_schema.to.1,
            properties: {
                match edge_schema.properties {
                    Some(properties) => Some(
                        properties
                            .into_iter()
                            .map(|p| Field {
                                name: p.name,
                                field_type: p.field_type,
                            })
                            .collect(),
                    ),
                    None => None,
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: FieldType,
}

impl From<ParserField> for Field {
    fn from(field: ParserField) -> Self {
        Field {
            name: field.name,
            field_type: field.field_type,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Query {
    pub original_query: String,
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub statements: Vec<Statement>,
    pub return_values: Vec<Expression>,
}

impl From<ParserQuery> for Query {
    fn from(query: ParserQuery) -> Self {
        Query {
            original_query: query.original_query,
            name: query.name,
            parameters: query
                .parameters
                .into_iter()
                .map(|p| Parameter {
                    name: p.name.1,
                    param_type: p.param_type.1,
                })
                .collect(),
            statements: query.statements.into_iter().map(Statement::from).collect(),
            return_values: query
                .return_values
                .into_iter()
                .map(Expression::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: FieldType,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assignment(Assignment),
    AddVector(AddVector),
    AddNode(AddNode),
    AddEdge(AddEdge),
    Drop(Expression),
    SearchVector(SearchVector),
    BatchAddVector(BatchAddVector),
    ForLoop(ForLoop),
}

impl From<ParserStatement> for Statement {
    fn from(statement: ParserStatement) -> Self {
        match statement.statement {
            ParserStatementType::Assignment(assignment) => Statement::Assignment(Assignment {
                variable: assignment.variable,
                value: Expression::from(assignment.value),
            }),
            ParserStatementType::AddVector(add_vector) => {
                Statement::AddVector(AddVector::from(add_vector))
            }
            ParserStatementType::AddNode(add_node) => Statement::AddNode(AddNode::from(add_node)),
            ParserStatementType::AddEdge(add_edge) => Statement::AddEdge(AddEdge::from(add_edge)),
            ParserStatementType::Drop(expr) => Statement::Drop(Expression::from(expr)),
            ParserStatementType::SearchVector(search) => {
                Statement::SearchVector(SearchVector::from(search))
            }
            ParserStatementType::BatchAddVector(batch) => {
                Statement::BatchAddVector(BatchAddVector::from(batch))
            }
            ParserStatementType::ForLoop(loop_) => Statement::ForLoop(ForLoop {
                variables: loop_.variables,
                in_variable: loop_.in_variable,
                statements: loop_.statements.into_iter().map(Statement::from).collect(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub variable: String,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct ForLoop {
    pub variables: Vec<String>,
    pub in_variable: String,
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Traversal(Box<Traversal>),
    Identifier(String),
    StringLiteral(String),
    IntegerLiteral(i32),
    FloatLiteral(f64),
    BooleanLiteral(bool),
    Exists(Box<Traversal>),
    BatchAddVector(BatchAddVector),
    AddVector(AddVector),
    AddNode(AddNode),
    AddEdge(AddEdge),
    And(Vec<Expression>),
    Or(Vec<Expression>),
    SearchVector(SearchVector),
    Empty,
}

impl From<ParserExpression> for Expression {
    fn from(expression: ParserExpression) -> Self {
        match expression.expr {
            ParserExpressionType::Traversal(traversal) => {
                Expression::Traversal(Box::new(Traversal::from(*traversal)))
            }
            ParserExpressionType::Identifier(s) => Expression::Identifier(s),
            ParserExpressionType::StringLiteral(s) => Expression::StringLiteral(s),
            ParserExpressionType::IntegerLiteral(i) => Expression::IntegerLiteral(i),
            ParserExpressionType::FloatLiteral(f) => Expression::FloatLiteral(f),
            ParserExpressionType::BooleanLiteral(b) => Expression::BooleanLiteral(b),
            ParserExpressionType::Exists(t) => Expression::Exists(Box::new(Traversal::from(*t))),
            ParserExpressionType::BatchAddVector(b) => {
                Expression::BatchAddVector(BatchAddVector::from(b))
            }
            ParserExpressionType::AddVector(v) => Expression::AddVector(AddVector::from(v)),
            ParserExpressionType::AddNode(n) => Expression::AddNode(AddNode::from(n)),
            ParserExpressionType::AddEdge(e) => Expression::AddEdge(AddEdge::from(e)),
            ParserExpressionType::And(exprs) => {
                Expression::And(exprs.into_iter().map(Expression::from).collect())
            }
            ParserExpressionType::Or(exprs) => {
                Expression::Or(exprs.into_iter().map(Expression::from).collect())
            }
            ParserExpressionType::SearchVector(s) => {
                Expression::SearchVector(SearchVector::from(s))
            }
            ParserExpressionType::Empty => Expression::Empty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Traversal {
    pub start: StartNode,
    pub steps: Vec<Step>,
}

impl From<ParserTraversal> for Traversal {
    fn from(traversal: ParserTraversal) -> Self {
        Traversal {
            start: StartNode::from(traversal.start),
            steps: traversal.steps.into_iter().map(Step::from).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchAddVector {
    pub vector_type: Option<String>,
    pub vec_identifier: Option<String>,
    pub fields: Option<HashMap<String, ValueType>>,
}

impl From<ParserBatchAddVector> for BatchAddVector {
    fn from(batch_add_vector: ParserBatchAddVector) -> Self {
        BatchAddVector {
            vector_type: batch_add_vector.vector_type,
            vec_identifier: batch_add_vector.vec_identifier,
            fields: batch_add_vector.fields.map(|f| {
                f.into_iter()
                    .map(|(k, v)| (k, ValueType::from(v)))
                    .collect()
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StartNode {
    Node {
        types: Option<Vec<String>>,
        ids: Option<Vec<String>>,
    },
    Edge {
        types: Option<Vec<String>>,
        ids: Option<Vec<String>>,
    },
    Variable(String),
    Anonymous,
}

impl From<ParserStartNode> for StartNode {
    fn from(start_node: ParserStartNode) -> Self {
        match start_node {
            ParserStartNode::Node { types, ids } => StartNode::Node { types, ids },
            ParserStartNode::Edge { types, ids } => StartNode::Edge { types, ids },
            ParserStartNode::Variable(s) => StartNode::Variable(s),
            ParserStartNode::Anonymous => StartNode::Anonymous,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Step {
    Node(GraphStep),
    Edge(GraphStep),
    Where(Box<Expression>),
    BooleanOperation(BooleanOp),
    Count,
    Update(Update),
    Object(Object),
    Exclude(Exclude),
    Closure(Closure),
    Range((Expression, Expression)),
    AddEdge(AddEdge),
    SearchVector(String),
}

impl From<ParserStep> for Step {
    fn from(step: ParserStep) -> Self {
        match step.step {
            ParserStepType::Node(g) => Step::Node(GraphStep::from(g)),
            ParserStepType::Edge(g) => Step::Edge(GraphStep::from(g)),
            ParserStepType::Where(e) => Step::Where(Box::new(Expression::from(*e))),
            ParserStepType::BooleanOperation(b) => Step::BooleanOperation(BooleanOp::from(b)),
            ParserStepType::Count => Step::Count,
            ParserStepType::Update(u) => Step::Update(Update::from(u)),
            ParserStepType::Object(o) => Step::Object(Object::from(o)),
            ParserStepType::Exclude(e) => Step::Exclude(Exclude::from(e)),
            ParserStepType::Closure(c) => Step::Closure(Closure::from(c)),
            ParserStepType::Range(r) => Step::Range((Expression::from(r.0), Expression::from(r.1))),
            ParserStepType::AddEdge(e) => Step::AddEdge(AddEdge::from(e)),
            ParserStepType::SearchVector(s) => Step::SearchVector(s),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GraphStep {
    Out(Option<Vec<String>>),
    In(Option<Vec<String>>),
    FromN,
    ToN,
    OutE(Option<Vec<String>>),
    InE(Option<Vec<String>>),
}

impl From<ParserGraphStep> for GraphStep {
    fn from(step: ParserGraphStep) -> Self {
        match step.step {
            ParserGraphStepType::Out(v) => GraphStep::Out(v),
            ParserGraphStepType::In(v) => GraphStep::In(v),
            ParserGraphStepType::FromN => GraphStep::FromN,
            ParserGraphStepType::ToN => GraphStep::ToN,
            ParserGraphStepType::OutE(v) => GraphStep::OutE(v),
            ParserGraphStepType::InE(v) => GraphStep::InE(v),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BooleanOp {
    And(Vec<Expression>),
    Or(Vec<Expression>),
    GreaterThan(Box<Expression>),
    GreaterThanOrEqual(Box<Expression>),
    LessThan(Box<Expression>),
    LessThanOrEqual(Box<Expression>),
    Equal(Box<Expression>),
    NotEqual(Box<Expression>),
}

impl From<ParserBooleanOp> for BooleanOp {
    fn from(op_type: ParserBooleanOp) -> Self {
        match op_type.op {
            ParserBooleanOpType::And(exprs) => {
                BooleanOp::And(exprs.into_iter().map(Expression::from).collect())
            }
            ParserBooleanOpType::Or(exprs) => {
                BooleanOp::Or(exprs.into_iter().map(Expression::from).collect())
            }
            ParserBooleanOpType::GreaterThan(e) => {
                BooleanOp::GreaterThan(Box::new(Expression::from(*e)))
            }
            ParserBooleanOpType::GreaterThanOrEqual(e) => {
                BooleanOp::GreaterThanOrEqual(Box::new(Expression::from(*e)))
            }
            ParserBooleanOpType::LessThan(e) => BooleanOp::LessThan(Box::new(Expression::from(*e))),
            ParserBooleanOpType::LessThanOrEqual(e) => {
                BooleanOp::LessThanOrEqual(Box::new(Expression::from(*e)))
            }
            ParserBooleanOpType::Equal(e) => BooleanOp::Equal(Box::new(Expression::from(*e))),
            ParserBooleanOpType::NotEqual(e) => BooleanOp::NotEqual(Box::new(Expression::from(*e))),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VectorData {
    Vector(Vec<f64>),
    Identifier(String),
}

impl From<ParserVectorData> for VectorData {
    fn from(data: ParserVectorData) -> Self {
        match data {
            ParserVectorData::Vector(v) => VectorData::Vector(v),
            ParserVectorData::Identifier(s) => VectorData::Identifier(s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchVector {
    pub vector_type: Option<String>,
    pub data: Option<VectorData>,
    pub k: Option<EvaluatesToNumber>,
}

impl From<ParserSearchVector> for SearchVector {
    fn from(search_vector: ParserSearchVector) -> Self {
        SearchVector {
            vector_type: search_vector.vector_type,
            data: search_vector.data.map(VectorData::from),
            k: search_vector.k.map(EvaluatesToNumber::from),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EvaluatesToNumber {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
    Identifier(String),
}

impl From<ParserEvaluatesToNumber> for EvaluatesToNumber {
    fn from(num_type: ParserEvaluatesToNumber) -> Self {
        match num_type.value {
            ParserEvaluatesToNumberType::I8(n) => EvaluatesToNumber::I8(n),
            ParserEvaluatesToNumberType::I16(n) => EvaluatesToNumber::I16(n),
            ParserEvaluatesToNumberType::I32(n) => EvaluatesToNumber::I32(n),
            ParserEvaluatesToNumberType::I64(n) => EvaluatesToNumber::I64(n),
            ParserEvaluatesToNumberType::U8(n) => EvaluatesToNumber::U8(n),
            ParserEvaluatesToNumberType::U16(n) => EvaluatesToNumber::U16(n),
            ParserEvaluatesToNumberType::U32(n) => EvaluatesToNumber::U32(n),
            ParserEvaluatesToNumberType::U64(n) => EvaluatesToNumber::U64(n),
            ParserEvaluatesToNumberType::U128(n) => EvaluatesToNumber::U128(n),
            ParserEvaluatesToNumberType::F32(n) => EvaluatesToNumber::F32(n),
            ParserEvaluatesToNumberType::F64(n) => EvaluatesToNumber::F64(n),
            ParserEvaluatesToNumberType::Identifier(s) => EvaluatesToNumber::Identifier(s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddVector {
    pub vector_type: Option<String>,
    pub data: Option<VectorData>,
    pub fields: Option<HashMap<String, ValueType>>,
}

impl From<ParserAddVector> for AddVector {
    fn from(add_vector: ParserAddVector) -> Self {
        AddVector {
            vector_type: add_vector.vector_type,
            data: add_vector.data.map(VectorData::from),
            fields: add_vector.fields.map(|f| {
                f.into_iter()
                    .map(|(k, v)| (k, ValueType::from(v)))
                    .collect()
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddNode {
    pub node_type: Option<String>,
    pub fields: Option<HashMap<String, ValueType>>,
}

impl From<ParserAddNode> for AddNode {
    fn from(add_node: ParserAddNode) -> Self {
        AddNode {
            node_type: add_node.node_type,
            fields: add_node.fields.map(|f| {
                f.into_iter()
                    .map(|(k, v)| (k, ValueType::from(v)))
                    .collect()
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddEdge {
    pub edge_type: Option<String>,
    pub fields: Option<HashMap<String, ValueType>>,
    pub connection: EdgeConnection,
    pub from_identifier: bool,
}

impl From<ParserAddEdge> for AddEdge {
    fn from(add_edge: ParserAddEdge) -> Self {
        AddEdge {
            edge_type: add_edge.edge_type,
            fields: add_edge.fields.map(|f| {
                f.into_iter()
                    .map(|(k, v)| (k, ValueType::from(v)))
                    .collect()
            }),
            connection: EdgeConnection::from(add_edge.connection),
            from_identifier: add_edge.from_identifier,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdgeConnection {
    pub from_id: Option<IdType>,
    pub to_id: Option<IdType>,
}

impl From<ParserEdgeConnection> for EdgeConnection {
    fn from(edge_connection: ParserEdgeConnection) -> Self {
        EdgeConnection {
            from_id: edge_connection.from_id.map(IdType::from),
            to_id: edge_connection.to_id.map(IdType::from),
        }
    }
}

#[derive(Debug, Clone)]
pub enum IdType {
    Literal(String),
    Identifier(String),
}

impl From<ParserIdType> for IdType {
    fn from(id_type: ParserIdType) -> Self {
        match id_type {
            ParserIdType::Literal(s) => IdType::Literal(s),
            ParserIdType::Identifier(s) => IdType::Identifier(s),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueType {
    Literal(Value),
    Identifier(String),
    Object(Object),
}

impl From<ParserValueType> for ValueType {
    fn from(value_type: ParserValueType) -> Self {
        match value_type {
            ParserValueType::Literal(v) => ValueType::Literal(v),
            ParserValueType::Identifier(s) => ValueType::Identifier(s),
            ParserValueType::Object(o) => ValueType::Object(Object::from(o)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Update {
    pub fields: Vec<FieldAddition>,
}

impl From<ParserUpdate> for Update {
    fn from(update: ParserUpdate) -> Self {
        Update {
            fields: update.fields.into_iter().map(FieldAddition::from).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Object {
    pub fields: Vec<(String, FieldValue)>,
    pub should_spread: bool,
}

impl From<ParserObject> for Object {
    fn from(object: ParserObject) -> Self {
        Object {
            fields: object
                .fields
                .into_iter()
                .map(|(k, v)| (k, FieldValue::from(v)))
                .collect(),
            should_spread: object.should_spread,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Exclude {
    pub fields: Vec<String>,
}

impl From<ParserExclude> for Exclude {
    fn from(exclude: ParserExclude) -> Self {
        Exclude {
            fields: exclude.fields.into_iter().map(|(_, s)| s).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub identifier: String,
    pub object: Object,
}

impl From<ParserClosure> for Closure {
    fn from(closure: ParserClosure) -> Self {
        Closure {
            identifier: closure.identifier,
            object: Object::from(closure.object),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldAddition {
    pub name: String,
    pub value: FieldValue,
}

impl From<ParserFieldAddition> for FieldAddition {
    fn from(field_addition: ParserFieldAddition) -> Self {
        FieldAddition {
            name: field_addition.name,
            value: FieldValue::from(field_addition.value),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FieldValue {
    Traversal(Box<Traversal>),
    Expression(Expression),
    Fields(Vec<FieldAddition>),
    Literal(Value),
    Identifier(String),
    Empty,
}

impl From<ParserFieldValue> for FieldValue {
    fn from(field_value_type: ParserFieldValue) -> Self {
        match field_value_type.value {
            ParserFieldValueType::Traversal(t) => {
                FieldValue::Traversal(Box::new(Traversal::from(*t)))
            }
            ParserFieldValueType::Expression(e) => FieldValue::Expression(Expression::from(e)),
            ParserFieldValueType::Fields(f) => {
                FieldValue::Fields(f.into_iter().map(FieldAddition::from).collect())
            }
            ParserFieldValueType::Literal(v) => FieldValue::Literal(v),
            ParserFieldValueType::Empty => FieldValue::Empty,
            ParserFieldValueType::Identifier(s) => FieldValue::Identifier(s),
        }
    }
}
