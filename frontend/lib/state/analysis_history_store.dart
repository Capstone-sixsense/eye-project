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
}
