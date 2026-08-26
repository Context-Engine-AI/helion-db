// mod tests {
//     use super::*;
//     use crate::helixc::{generator::generator::{to_snake_case, CodeGenerator}, parser::helix_parser::HelixParser};
//     use pest::Parser;

//     #[test]
//     fn test_basic_query_generation() {
//         let input = r#"
//         QUERY GetUser(id: String) =>
//             user <- V("id")
//             RETURN user
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let output = generator.generate_source(&source);
//         println!("{}", output);
//         assert!(output.contains("pub fn get_user"));
//         assert!(output.contains("struct GetUserData"));
//         assert!(output.contains("id: String"));
//     }

//     #[test]
//     fn test_add_node_generation() {
//         let input = r#"
//         QUERY CreateUser(name: String, age: I32) =>
//             user <- AddN<User>({Name: "name", Age: "age"})
//             RETURN user
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let output = generator.generate_source(&source);

//         assert!(output.contains("add_n"));
//         assert!(output.contains("props!"));
//         assert!(output.contains("Name"));
//         assert!(output.contains("Age"));
//     }

//     #[test]
//     fn test_where_simple_condition() {
//         let input = r#"
//         QUERY FindActiveUsers() =>
//             users <- V<User>::WHERE(_::{is_enabled}::EQ(true))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);
//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("is_enabled"));
//         assert!(generated.contains("=="));
//     }

//     #[test]
//     fn test_where_exists_condition() {
//         let input = r#"
//         QUERY FindUsersWithPosts() =>
//             users <- V<User>::WHERE(EXISTS(_::OutE<Authored>))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         println!("Source:\n{:?}", source);
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);

//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("out_e"));
//         assert!(generated.contains("count"));
//         assert!(generated.contains("count > 0"));
//     }

//     #[test]
//     fn test_where_and_condition() {
//         let input = r#"
//         QUERY FindVerifiedActiveUsers() =>
//             users <- V<User>::WHERE(AND(
//                 _::{verified}::EQ(true),
//                 _::{is_enabled}::EQ(true)
//             ))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         println!("Source:\n{:?}", source);
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);
//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("&&"));
//         assert!(generated.contains("verified"));
//         assert!(generated.contains("is_enabled"));
//     }

//     #[test]
//     fn test_where_or_condition() {
//         let input = r#"
//         QUERY FindSpecialUsers() =>
//             users <- V<User>::WHERE(OR(
//                 _::{verified}::EQ(true),
//                 _::{followers_count}::GT(1000)
//             ))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);

//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("||"));
//         assert!(generated.contains("verified"));
//         assert!(generated.contains("followers_count"));
//     }

//     #[test]
//     fn test_where_complex_traversal() {
//         let input = r#"
//         QUERY FindInfluentialUsers() =>
//             users <- V<User>::WHERE(
//                 _::Out<Follows>::COUNT::GT(100)
//             )::WHERE(
//                 _::In<Follows>::COUNT::GT(1000)
//             )
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);

//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("out"));
//         assert!(generated.contains("in_"));
//         assert!(generated.contains("count"));
//         assert!(generated.contains(">"));
//     }

//     #[test]
//     fn test_where_with_nested_conditions() {
//         let input = r#"
//         QUERY FindComplexUsers() =>
//             users <- V<User>::WHERE(AND(
//                 OR(
//                     _::{verified}::EQ(true),
//                     _::{followers_count}::GT(5000)
//                 ),
//                 _::Out<Authored>::COUNT::GT(10)
//             ))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         println!("Source:\n{:?}", source);
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);
//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("&&"));
//         assert!(generated.contains("||"));
//         assert!(generated.contains("verified"));
//         assert!(generated.contains("followers_count"));
//         assert!(generated.contains("out"));
//         assert!(generated.contains("count"));
//     }

//     #[test]
//     fn test_boolean_operations() {
//         let input = r#"
//         QUERY FindUsersWithSpecificProperty(property_name: String, value: String) =>
//             users <- V<User>::WHERE(_::{property_name}::EQ(value))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);

//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("property_name"));
//         assert!(generated.contains("=="));
//         assert!(generated.contains("value"));
//         assert!(generated.contains("check_property"));
//     }

//     #[test]
//     fn test_boolean_operations_with_multiple_properties() {
//         let input = r#"
//         QUERY FindUsersWithSpecificProperties(property1: String, value1: String, property2: String, value2: String, property3: String, value3: String) =>
//             users <- V<User>::WHERE(AND(
//                 _::{property1}::EQ(value1),
//                 _::{property2}::EQ(value2),
//                 _::{property3}::EQ(value3)
//             ))
//             RETURN users
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("Generated code:\n{}", generated);

//         assert!(generated.contains("filter_ref"));
//         assert!(generated.contains("&&"));
//         assert!(generated.contains("property1"));
//         assert!(generated.contains("property2"));
//         assert!(generated.contains("property3"));
//         assert!(generated.contains("value1"));
//         assert!(generated.contains("value2"));
//         assert!(generated.contains("value3"));
//     }

//     #[test]
//     fn test_to_snake_case() {
//         assert_eq!(to_snake_case("camelCase"), "camel_case");
//         assert_eq!(to_snake_case("UserIds"), "user_ids");
//         assert_eq!(to_snake_case("SimpleXMLParser"), "simple_xml_parser");
//         assert_eq!(to_snake_case("type"), "type_");
//         assert_eq!(to_snake_case("ID"), "id");
//         assert_eq!(to_snake_case("UserID"), "user_id");
//         assert_eq!(to_snake_case("XMLHttpRequest"), "xml_http_request");
//         assert_eq!(to_snake_case("iOS"), "i_os");
//     }
// }
