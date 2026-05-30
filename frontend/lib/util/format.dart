/// 로컬 시각을 `YYYY-MM-DD HH:mm` 또는 `YYYY-MM-DD HH:mm:ss` 형식으로 표시.
String formatLocalDateTime(DateTime value, {bool includeSeconds = false}) {
  final local = value.toLocal();
  final y = local.year.toString().padLeft(4, '0');
  final m = local.month.toString().padLeft(2, '0');
  final d = local.day.toString().padLeft(2, '0');
  final h = local.hour.toString().padLeft(2, '0');
  final mi = local.minute.toString().padLeft(2, '0');
  if (!includeSeconds) return '$y-$m-$d $h:$mi';
  final s = local.second.toString().padLeft(2, '0');
  return '$y-$m-$d $h:$mi:$s';
}

/// ISO 8601 문자열을 로컬 시각 문자열로 변환. 파싱 실패 시 원문 반환.
String formatIsoTimestamp(String ts, {bool includeSeconds = false}) {
  final parsed = DateTime.tryParse(ts);
  if (parsed == null) return ts;
  return formatLocalDateTime(parsed, includeSeconds: includeSeconds);
}
