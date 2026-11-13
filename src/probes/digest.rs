use sha2::{Digest, Sha256};

/// Deterministic, zero-LLM tool output compressor for System-1.
/// Guarantees every `sensor_digest` stays within `max_chars` (~4 chars/token).
pub fn compress_tool_output(
    tool_name: &str,
    exit_code: i32,
    raw_output: &str,
    max_tokens: usize,
) -> String {
    let max_chars = max_tokens.saturating_mul(4).max(160);
    let digest = if tool_name.contains("sql") || tool_name.contains("db") {
        compress_sql(raw_output, exit_code)
    } else if tool_name.contains("patch") || tool_name.contains("edit") || raw_output.contains("@@")
    {
        compress_diff(raw_output, exit_code)
    } else {
        compress_shell_or_text(raw_output, exit_code)
    };

    if digest.len() <= max_chars {
        digest
    } else {
        let mut cut = max_chars.saturating_sub(3);
        while !digest.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        format!("{}...", &digest[..cut])
    }
}

fn short_hash(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let res = hasher.finalize();
    format!("{:02x}{:02x}{:02x}{:02x}", res[0], res[1], res[2], res[3])
}

fn compress_diff(raw: &str, exit_code: i32) -> String {
    let mut added = 0usize;
    let mut removed = 0usize;
    let mut files = Vec::new();
    for line in raw.lines() {
        if let Some(f) = line.strip_prefix("+++ b/").or_else(|| line.strip_prefix("+++ ")) {
            if files.len() < 3 {
                files.push(f.trim());
            }
        } else if line.starts_with('+') && !line.starts_with("+++") {
            added += 1;
        } else if line.starts_with('-') && !line.starts_with("---") {
            removed += 1;
        }
    }
    format!(
        "status={} files=[{}] +{} -{} sha={}",
        if exit_code == 0 { "OK" } else { "ERR" },
        files.join(","),
        added,
        removed,
        short_hash(raw)
    )
}

fn compress_sql(raw: &str, exit_code: i32) -> String {
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    let row_count = lines.len().saturating_sub(1);
    let preview: Vec<String> = lines
        .iter()
        .take(3)
        .map(|l| {
            if l.len() > 80 {
                format!("{}..", &l[..78])
            } else {
                l.to_string()
            }
        })
        .collect();
    format!(
        "sql_exit={} rows={} preview=[{}]",
        exit_code,
        row_count,
        preview.join(" | ")
    )
}

fn compress_shell_or_text(raw: &str, exit_code: i32) -> String {
    let mut anchors = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.contains("error[")
            || trimmed.contains("FAILED")
            || trimmed.contains("panicked at")
            || trimmed.contains("Error:")
            || trimmed.contains("Exception")
        {
            anchors.push(trimmed);
            if anchors.len() >= 3 {
                break;
            }
        }
    }
    if !anchors.is_empty() {
        return format!("exit={} anchors=[{}]", exit_code, anchors.join(" ; "));
    }
    let compact: String = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" / ");
    format!("exit={} out=[{}]", exit_code, compact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_caps_output_and_extracts_anchors() {
        let long_err = format!(
            "compiling...\nerror[E0308]: mismatched types at src/stats/bocpd.rs:42\n{}",
            "x".repeat(2000)
        );
        let d = compress_tool_output("cargo_test", 101, &long_err, 120);
        assert!(d.len() <= 480);
        assert!(d.contains("exit=101"));
        assert!(d.contains("error[E0308]"));
    }
}
