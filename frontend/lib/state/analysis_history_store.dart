import '../models/analysis_history_entry.dart';

class AnalysisHistoryStore {
  AnalysisHistoryStore._();

  static final List<AnalysisHistoryEntry> _entries = <AnalysisHistoryEntry>[];

  static List<AnalysisHistoryEntry> get entries =>
      List<AnalysisHistoryEntry>.unmodifiable(_entries);

  static void add(AnalysisHistoryEntry entry) {
    _entries.insert(0, entry);
  }

  static void clear() {
    _entries.clear();
  }

  /// [indices]에 해당하는 항목만 제거한다. 같은 인덱스가 두 번 들어오면 한 번만 제거된다.
  static void removeAtIndices(Iterable<int> indices) {
    final sortedDesc = indices.toSet().toList()
      ..sort((a, b) => b.compareTo(a));
    for (final i in sortedDesc) {
      if (i >= 0 && i < _entries.length) {
        _entries.removeAt(i);
      }
    }
  }
}
