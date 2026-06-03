import 'package:flutter/material.dart';

import '../api/eye_api_client.dart';
import '../models/server_log_entry.dart';
import '../util/format.dart';
import 'dialog_keyboard.dart';
import 'medical_ui.dart';
import 'notice_dialog.dart' show showErrorNotice;

/// 서버 분석 로그 조회 다이얼로그.
Future<void> showServerLogsDialog(BuildContext context) {
  final width = MediaQuery.sizeOf(context).width;
  final dialogWidth = (width - 48).clamp(320.0, 640.0);

  return showDialog<void>(
    context: context,
    barrierDismissible: true,
    builder: (dialogContext) {
      void close() => Navigator.of(dialogContext).pop();
      return dialogOkShortcuts(
        onClose: close,
        child: Dialog(
          child: ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: dialogWidth,
              maxHeight: MediaQuery.sizeOf(dialogContext).height * 0.82,
            ),
            child: const _ServerLogsDialogBody(),
          ),
        ),
      );
    },
  );
}

class _ServerLogsDialogBody extends StatefulWidget {
  const _ServerLogsDialogBody();

  @override
  State<_ServerLogsDialogBody> createState() => _ServerLogsDialogBodyState();
}

class _ServerLogsDialogBodyState extends State<_ServerLogsDialogBody> {
  static const int _pageLimit = 50;

  final EyeApiClient _api = EyeApiClient();
  final ScrollController _scrollController = ScrollController();

  final List<ServerLogEntry> _items = [];
  int _total = 0;
  bool _loading = true;
  bool _loadingMore = false;
  String? _error;
  String? _levelFilter;

  bool get _hasMore => _items.length < _total;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    _loadInitial();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _api.close();
    super.dispose();
  }

  void _onScroll() {
    if (!_hasMore || _loadingMore || _loading) return;
    final pos = _scrollController.position;
    if (pos.pixels >= pos.maxScrollExtent - 120) {
      _loadMore();
    }
  }

  Future<void> _loadInitial() async {
    setState(() {
      _loading = true;
      _error = null;
      _items.clear();
      _total = 0;
    });
    try {
      final page = await _api.fetchLogsPage(
        limit: _pageLimit,
        offset: 0,
        level: _levelFilter,
      );
      if (!mounted) return;
      setState(() {
        _items.addAll(page.items);
        _total = page.total;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _loadMore() async {
    if (_loadingMore || !_hasMore) return;
    setState(() => _loadingMore = true);
    try {
      final page = await _api.fetchLogsPage(
        limit: _pageLimit,
        offset: _items.length,
        level: _levelFilter,
      );
      if (!mounted) return;
      setState(() {
        _items.addAll(page.items);
        _total = page.total;
        _loadingMore = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loadingMore = false);
      await showErrorNotice(context, e);
    }
  }

  void _toggleLevelFilter(String level) {
    setState(() {
      _levelFilter = _levelFilter == level ? null : level;
    });
    if (_scrollController.hasClients) {
      _scrollController.jumpTo(0);
    }
    _loadInitial();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    void close() => Navigator.of(context).pop();

    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 8, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '서버 로그',
                      style: theme.textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: MedicalTokens.textMain,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '분석·서버 동작 기록',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: MedicalTokens.textSubtle,
                        height: 1.45,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(onPressed: close, icon: const Icon(Icons.close)),
            ],
          ),
          const SizedBox(height: 8),
          const MedicalNoticeBanner(
            body: '로그는 1개월 뒤 자동으로 삭제됩니다.',
          ),
          const SizedBox(height: 10),
          _LogLevelFilterBar(
            activeLevel: _levelFilter,
            onToggle: _toggleLevelFilter,
          ),
          const SizedBox(height: 12),
          Expanded(child: _buildLogList(theme)),
        ],
      ),
    );
  }

  Widget _buildLogList(ThemeData theme) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '로그를 불러오지 못했습니다.',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: MedicalTokens.textSubtle,
              ),
            ),
            const SizedBox(height: 12),
            MedicalSecondaryButton(
              label: '다시 시도',
              onPressed: _loadInitial,
            ),
          ],
        ),
      );
    }
    if (_items.isEmpty) {
      return Center(
        child: Text(
          _levelFilter != null ? '해당 레벨의 로그가 없습니다.' : '저장된 로그가 없습니다.',
          style: theme.textTheme.bodyMedium?.copyWith(
            color: MedicalTokens.textSubtle,
          ),
        ),
      );
    }

    return Scrollbar(
      controller: _scrollController,
      thumbVisibility: true,
      child: ListView.separated(
        controller: _scrollController,
        itemCount: _items.length + (_loadingMore ? 1 : 0),
        separatorBuilder: (_, _) => const SizedBox(height: 8),
        itemBuilder: (context, index) {
          if (index >= _items.length) {
            return const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Center(
                child: SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
            );
          }
          return _LogRow(entry: _items[index]);
        },
      ),
    );
  }
}

({Color background, Color foreground}) _logLevelStyle(String level) {
  switch (level.toUpperCase()) {
    case 'WARNING':
      return (
        background: MedicalTokens.primarySoft,
        foreground: const Color(0xFF1F5F7A),
      );
    case 'ERROR':
      return (
        background: const Color(0xFFFDE8E8),
        foreground: const Color(0xFFB42318),
      );
    default:
      return (
        background: const Color(0xFFF0F4F8),
        foreground: MedicalTokens.textMain,
      );
  }
}

class _LogLevelFilterBar extends StatelessWidget {
  const _LogLevelFilterBar({
    required this.activeLevel,
    required this.onToggle,
  });

  final String? activeLevel;
  final ValueChanged<String> onToggle;

  static const _levels = ['INFO', 'WARNING', 'ERROR'];

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (var i = 0; i < _levels.length; i++) ...[
          if (i > 0) const SizedBox(width: 8),
          _LogLevelFilterChip(
            level: _levels[i],
            selected: activeLevel == _levels[i],
            onTap: () => onToggle(_levels[i]),
          ),
        ],
      ],
    );
  }
}

class _LogLevelFilterChip extends StatelessWidget {
  const _LogLevelFilterChip({
    required this.level,
    required this.selected,
    required this.onTap,
  });

  final String level;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final style = _logLevelStyle(level);
    final theme = Theme.of(context);

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(999),
        child: Ink(
          decoration: BoxDecoration(
            color: style.background,
            borderRadius: BorderRadius.circular(999),
            border: selected
                ? Border.all(color: style.foreground, width: 1.5)
                : null,
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            child: Text(
              level,
              style: theme.textTheme.labelMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: style.foreground,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _LogRow extends StatelessWidget {
  const _LogRow({required this.entry});

  final ServerLogEntry entry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final levelStyle = _logLevelStyle(entry.level);

    return MedicalCard(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Text(
                  formatIsoTimestamp(entry.ts, includeSeconds: true),
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: MedicalTokens.textSubtle,
                  ),
                ),
              ),
              MedicalBadge(
                text: entry.level,
                backgroundColor: levelStyle.background,
                foregroundColor: levelStyle.foreground,
              ),
              if (entry.phaseLabel != null) ...[
                const SizedBox(width: 6),
                MedicalBadge(
                  text: entry.phaseLabel!,
                  backgroundColor: MedicalTokens.primarySoft,
                  foregroundColor: MedicalTokens.textMain,
                ),
              ],
            ],
          ),
          const SizedBox(height: 6),
          Text(
            entry.message,
            style: theme.textTheme.bodySmall?.copyWith(
              color: MedicalTokens.textMain,
              height: 1.45,
            ),
          ),
          if (entry.jobId != null || entry.elapsed != null) ...[
            const SizedBox(height: 6),
            Text(
              [
                if (entry.jobId != null) 'job: ${entry.jobId}',
                if (entry.elapsed != null)
                  '${entry.elapsed!.toStringAsFixed(2)}s',
              ].join(' · '),
              style: theme.textTheme.labelSmall?.copyWith(
                color: MedicalTokens.textSubtle,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
