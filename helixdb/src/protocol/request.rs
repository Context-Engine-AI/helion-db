use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader, Result};

/// Default maximum request body size: 64 MiB.
/// Override with the `HELIX_MAX_BODY_MB` env var.
const DEFAULT_MAX_BODY_MB: usize = 64;

/// Maximum request body size in bytes. Reads from `HELIX_MAX_BODY_MB` on
/// first call and caches the result for the lifetime of the process.
pub fn max_body_bytes() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_MAX_BODY_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_BODY_MB)
            .saturating_mul(1024 * 1024)
    })
}

fn normalize_path(target: &str) -> String {
    let path = target
        .split_once('?')
        .map(|(path, _)| path)
        .unwrap_or(target);
    let path = path.split_once('#').map(|(path, _)| path).unwrap_or(path);

    if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct RequestHead {
    pub method: String,
    pub headers: HashMap<String, String>,
    pub path: String,
}

impl RequestHead {
    pub fn content_length(&self) -> Option<usize> {
        self.headers
            .get("content-length")
            .and_then(|length| length.parse::<usize>().ok())
    }

    pub async fn from_reader<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> Result<Self> {
        let mut first_line = String::new();
        reader.read_line(&mut first_line).await?;

        let mut parts = first_line.trim().split_whitespace();
        let method = parts
            .next()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Missing HTTP method: {}", first_line),
                )
            })?
            .to_string();
        let path = parts
            .next()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Missing path: {}", first_line),
                )
            })?
            .to_string();
        let path = normalize_path(&path);

        let mut headers = HashMap::new();
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await?;
            if bytes_read == 0 || line.eq("\r\n") || line.eq("\n") {
                break;
            }
            if let Some((key, value)) = line.trim().split_once(':') {
                headers.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
        }

        Ok(Self {
            method,
            headers,
            path,
        })
    }
}

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub headers: HashMap<String, String>,
    pub path: String,
    pub body: Vec<u8>,
}

impl Request {
    pub async fn from_reader<R: AsyncRead + Unpin>(
        reader: &mut BufReader<R>,
        head: RequestHead,
    ) -> Result<Request> {
        let mut body = Vec::new();
        if let Some(length) = head.content_length() {
            let cap = max_body_bytes();
            if length > cap {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::FileTooLarge,
                    format!("body {} bytes exceeds cap {} bytes", length, cap),
                ));
            }
            let mut buffer = vec![0; length];
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                reader.read_exact(&mut buffer),
            )
            .await
            {
                Ok(Ok(_)) => body = buffer,
                Ok(Err(e)) => {
                    return Err(std::io::Error::other(format!("Error reading body: {}", e)));
                }
                Err(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Timeout reading body",
                    ));
                }
            }
        }

        Ok(Request {
            method: head.method,
            headers: head.headers,
            path: head.path,
            body,
        })
    }

    /// Parse a request from a stream
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::Cursor;
    /// use helixdb::protocol::request::Request;
    ///
    /// let runtime = tokio::runtime::Runtime::new().unwrap();
    /// runtime.block_on(async {
    ///     let mut stream = Cursor::new(b"GET /test HTTP/1.1\r\n\r\n");
    ///     let request = Request::from_stream(&mut stream).await.unwrap();
    ///     assert_eq!(request.method, "GET");
    ///     assert_eq!(request.path, "/test");
    /// });
    /// ```
    pub async fn from_stream<R: AsyncRead + Unpin>(stream: &mut R) -> Result<Request> {
        let mut reader = BufReader::new(stream);
        let head = RequestHead::from_reader(&mut reader).await?;
        Request::from_reader(&mut reader, head).await
    }
}

#[cfg(test)]
mod tests {
    use super::Request;
    use std::io::Cursor;

    #[tokio::test]
    async fn strips_query_string_from_request_path() {
        let mut stream = Cursor::new(
            b"PUT /collections/test/points?wait=true HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
        );
        let request = Request::from_stream(&mut stream).await.unwrap();
        assert_eq!(request.method, "PUT");
        assert_eq!(request.path, "/collections/test/points");
    }

    #[tokio::test]
    async fn rejects_oversized_body_without_allocating() {
        // Advertised Content-Length far exceeds the 64 MiB default cap.
        // The server must reject via ErrorKind::FileTooLarge *before* any
        // large allocation or read attempt.
        let mut stream = Cursor::new(
            b"POST /collections HTTP/1.1\r\nContent-Length: 1099511627776\r\n\r\n".to_vec(),
        );
        let err = Request::from_stream(&mut stream)
            .await
            .expect_err("expected body-too-large rejection");
        assert_eq!(err.kind(), std::io::ErrorKind::FileTooLarge);
    }
}
