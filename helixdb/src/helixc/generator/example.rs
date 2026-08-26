// use crate::helixc::parser::helix_parser::HelixParser;
// use super::generator::CodeGenerator;

// pub fn example_code_generation() -> String {
//     // Example Helix QL query
//     let input = r#"
//     V::User {
//         Name: String,
//         Age: Integer,
//         Email: String
//     }

//     V::Post {
//         Content: String,
//         CreatedAt: String
//     }

//     E::Authored {
//         From: User,
//         To: Post,
//         Properties: {}
//     }

//     QUERY CreateUser(name: String, age: Integer, email: String) =>
//         user <- AddV<User>({
//             Name: name,
//             Age: age,
//             Email: email
//         })
//         RETURN user

//     QUERY GetUserPosts(userId: String) =>
//         user <- V(userId)
//         posts <- user::OutE<Authored>::InV
//         RETURN posts

//     QUERY CreatePost(userId: String, content: String) =>
//         user <- V(userId)
//         post <- AddV<Post>({
//             Content: content,
//             CreatedAt: "2024-02-19"
//         })
//         AddE<Authored>::From(user)::To(post)
//         RETURN post
//     "#;

//     // Parse the Helix QL query
//     let source = match HelixParser::parse_source(input) {
//         Ok(source) => source,
//         Err(e) => panic!("Failed to parse Helix QL: {:?}", e),
//     };
//     println!("Source {:?}", source);
//     // Generate Rust code
//     let mut generator = CodeGenerator::new();
//     let generated_code = generator.generate_source(&source);

//     // Print the generated code for inspection
//     println!("Generated Rust code:\n{}", generated_code);

//     generated_code
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_code_generation() {
//         let generated = example_code_generation();

//         // Verify schema generation
//         assert!(generated.contains("struct User"));
//         assert!(generated.contains("struct Post"));

//         // Verify query function generation
//         assert!(generated.contains("pub fn create_user"));
//         assert!(generated.contains("pub fn get_user_posts"));
//         assert!(generated.contains("pub fn create_post"));

//         // Verify field types
//         assert!(generated.contains("name: String"));
//         assert!(generated.contains("age: i32"));
//         assert!(generated.contains("email: String"));

//         // Verify traversal generation
//         assert!(generated.contains("tr.v_from_id"));
//         assert!(generated.contains("tr.out_e"));
//         assert!(generated.contains("tr.in_v"));

//         // Verify vertex/edge creation
//         assert!(generated.contains("tr.add_v"));
//         assert!(generated.contains("tr.add_e"));
//     }

//     #[test]
//     fn test_simple_query() {
//         let input = r#"
//         QUERY FindUser(userName: String) =>
//             user <- V<User>
//             RETURN user
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);

//         assert!(generated.contains("pub fn find_user"));
//         assert!(generated.contains("user_name: String"));
//         assert!(generated.contains("tr.v_from_types"));
//     }

//     #[test]
//     fn test_complex_traversal() {
//         let input = r#"
//         QUERY GetUserNetwork(userId: String) =>
//             user <- V(userId)
//             friends <- user::Out<Follows>::InV<User>
//             friendsOfFriends <- friends::Out<Follows>
//             RETURN friendsOfFriends
//         "#;

//         let source = HelixParser::parse_source(input).unwrap();
//         println!("Source {:?}", source);
//         let mut generator = CodeGenerator::new();
//         let generated = generator.generate_source(&source);
//         println!("{}", generated);
//         assert!(generated.contains("pub fn get_user_network"));
//         assert!(generated.contains("user_id: String"));
//         assert!(generated.contains("tr.v_from_id"));
//         assert!(generated.contains("tr.out"));
//         assert!(generated.contains("tr.in_v"));
//     }
// }
