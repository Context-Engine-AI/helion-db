use std::collections::HashMap;
use tokio::io::{AsyncWrite, AsyncWriteExt, Result};
#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Response {
    /// Create a new response
    pub fn new() -> Response {
        let mut headers = HashMap::new();
        // TODO: Change to use router config for headers and default routes
        headers.insert("Content-Type".to_string(), "text/plain".to_string());

        Response {
            status: 200,
            headers,
            body: Vec::new(),
        }
    }

    /// Send response back via stream
    ///
    /// # Example
    ///
    /// ```rust
    /// use helixdb::protocol::response::Response;
    /// use tokio::io::{duplex, AsyncReadExt};
    ///
    /// let runtime = tokio::runtime::Runtime::new().unwrap();
    /// runtime.block_on(async {
    ///     let mut response = Response::new();
    ///
    ///     response.status = 200;
    ///     response.body = b"Hello World".to_vec();
    ///
    ///     let (mut writer, mut reader) = duplex(1024);
    ///     response.send(&mut writer).await.unwrap();
    ///     drop(writer);
    ///
    ///     let mut data = Vec::new();
    ///     reader.read_to_end(&mut data).await.unwrap();
    ///     let data = String::from_utf8(data).unwrap();
    ///
    ///     assert!(data.contains("HTTP/1.1 200 OK"));
    ///     assert!(data.contains("Content-Length: 11"));
    ///     assert!(data.contains("Hello World"));
    /// });
    /// ```

    pub async fn send<W: AsyncWrite + Unpin>(&mut self, stream: &mut W) -> Result<()> {
        let status_message = match self.status {
            204 => "No Content",
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            409 => "Conflict",
            422 => "Unprocessable Entity",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            _ => "Unknown",
        };
        let mut writer = tokio::io::BufWriter::new(stream);

        // Write status line
        writer
            .write_all(format!("HTTP/1.1 {} {}\r\n", self.status, status_message).as_bytes())
            .await?;

        // Write headers
        for (header, value) in &self.headers {
            writer
                .write_all(format!("{}: {}\r\n", header, value).as_bytes())
                .await
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error writing header: {}", e),
                    )
                })?;
        }

        writer
            .write_all(format!("Content-Length: {}\r\n\r\n", self.body.len()).as_bytes())
            .await?;

        // Write body
        writer.write_all(&self.body).await?;
        writer.flush().await?;
        Ok(())
    }
}
