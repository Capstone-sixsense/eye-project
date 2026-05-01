import 'package:flutter/material.dart';

import '../models/analysis_history_entry.dart';
import '../models/result_screen_args.dart';
import '../state/analysis_history_store.dart';
import '../ui/medical_ui.dart';

/// 이력 보기 — 현재 세션에서 `/analyze` 성공 응답을 저장해 표시.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  HistoryViewMode _viewMode = HistoryViewMode.dashboard;

  bool _selectionMode = false;

  final Set<int> _selectedIndices = <int>{};

  void _exitSelectionMode() {
    setState(() {
      _selectionMode = false;
      _selectedIndices.clear();
    });
  }

  Future<void> _onTrashPressed() async {
    final entries = AnalysisHistoryStore.entries;
    if (entries.isEmpty) return;

    if (!_selectionMode) {
      setState(() => _selectionMode = true);
      return;
    }

    await _deleteSelected();
  }

  void _syncSelectionToEntryCount(int length) {
    _selectedIndices.removeWhere((i) => i < 0 || i >= length);
  }

  void _toggleIndex(int index) {
    setState(() {
      if (_selectedIndices.contains(index)) {
        _selectedIndices.remove(index);
      } else {
        _selectedIndices.add(index);
      }
    });
  }

  void _toggleSelectAll() {
    final entries = AnalysisHistoryStore.entries;
    setState(() {
      if (entries.isEmpty) return;
      if (_selectedIndices.length == entries.length) {
        _selectedIndices.clear();
      } else {
        _selectedIndices
          ..clear()
          ..addAll(List<int>.generate(entries.length, (i) => i));
      }
    });
  }

  Future<void> _deleteSelected() async {
    final entries = AnalysisHistoryStore.entries;
    if (entries.isEmpty) return;

    if (_selectedIndices.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('삭제할 항목을 선택해주세요.')),
      );
      return;
    }

    final count = _selectedIndices.length;
    final confirmed = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('선택 항목 삭제'),
            content: Text('선택한 $count건의 이력을 삭제할까요?\n삭제 후에는 되돌릴 수 없습니다.'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('취소'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('삭제'),
              ),
            ],
          ),
        ) ??
        false;

    if (!confirmed || !mounted) return;

    AnalysisHistoryStore.removeAtIndices(_selectedIndices);
    setState(() {
      _selectedIndices.clear();
      if (AnalysisHistoryStore.entries.isEmpty) {
        _selectionMode = false;
      }
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('선택한 이력을 삭제했습니다.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final entries = AnalysisHistoryStore.entries;
    _syncSelectionToEntryCount(entries.length);

    final allSelected =
        entries.isNotEmpty && _selectedIndices.length == entries.length;

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: _HistoryViewModeToggle(
          value: _viewMode,
          onChanged: (mode) {
            if (_viewMode == mode) return;
            setState(() => _viewMode = mode);
          },
        ),
        actions: [
          IconButton(
            onPressed: entries.isEmpty ? null : _onTrashPressed,
            icon: const Icon(Icons.delete_outline),
            tooltip:
                _selectionMode ? '선택 항목 삭제' : '삭제할 항목 선택',
          ),
          if (_selectionMode && entries.isNotEmpty) ...[
            IconButton(
              onPressed: _toggleSelectAll,
              icon: Icon(allSelected ? Icons.deselect : Icons.select_all),
              tooltip: allSelected ? '전체 선택 해제' : '전체 선택',
            ),
            IconButton(
              onPressed: _exitSelectionMode,
              icon: const Icon(Icons.close),
              tooltip: '선택 종료',
            ),
          ],
        ],
      ),
      body: entries.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(MedicalTokens.spaceLg),
                child: Text(
                  '현재 세션에 분석 이력이 없습니다.\n이미지를 업로드해 분석을 먼저 진행해주세요.',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : (_viewMode == HistoryViewMode.list
              ? _HistoryListView(
                  entries: entries,
                  selectionMode: _selectionMode,
                  selectedIndices: _selectedIndices,
                  onToggleSelection: _toggleIndex,
                )
              : _HistoryDashboardView(
                  entries: entries,
                  selectionMode: _selectionMode,
                  selectedIndices: _selectedIndices,
                  onToggleSelection: _toggleIndex,
                )),
    );
  }
}

enum HistoryViewMode { dashboard, list }

class _HistoryViewModeToggle extends StatelessWidget {
  const _HistoryViewModeToggle({
    required this.value,
    required this.onChanged,
  });

  final HistoryViewMode value;
  final ValueChanged<HistoryViewMode> onChanged;

  @override
  Widget build(BuildContext context) {
    final selectedColor = Theme.of(context).colorScheme.primary;
    final unselectedColor = Theme.of(context).colorScheme.onSurfaceVariant;
    final selectedBg = MedicalTokens.primarySoft;

    Widget button({
      required IconData icon,
      required HistoryViewMode mode,
      required String tooltip,
    }) {
      final selected = value == mode;
      return InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => onChanged(mode),
        child: Container(
          width: 40,
          height: 34,
          decoration: BoxDecoration(
            color: selected ? selectedBg : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
          ),
          alignment: Alignment.center,
          child: Icon(
            icon,
            size: 20,
            color: selected ? selectedColor : unselectedColor,
          ),
        ),
      );
    }

    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFFF6F7FB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: MedicalTokens.border),
      ),
      child: Padding(
        padding: const EdgeInsets.all(3),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Tooltip(
              message: '대시보드 보기',
              child: button(
                icon: Icons.grid_view_rounded,
                mode: HistoryViewMode.dashboard,
                tooltip: '대시보드 보기',
              ),
            ),
            const SizedBox(width: 4),
            Tooltip(
              message: '리스트 보기',
              child: button(
                icon: Icons.list_rounded,
                mode: HistoryViewMode.list,
                tooltip: '리스트 보기',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HistoryListView extends StatelessWidget {
  const _HistoryListView({
    required this.entries,
    required this.selectionMode,
    required this.selectedIndices,
    required this.onToggleSelection,
  });

  final List<AnalysisHistoryEntry> entries;
  final bool selectionMode;
  final Set<int> selectedIndices;
  final ValueChanged<int> onToggleSelection;

  @override
  Widget build(BuildContext context) {
    return ListView.separated(
      padding: const EdgeInsets.all(MedicalTokens.spaceMd),
      itemCount: entries.length,
      separatorBuilder: (_, _) => const SizedBox(height: MedicalTokens.spaceSm),
      itemBuilder: (context, index) {
        final item = entries[index];
        final subtitle = item.response.label ?? '판정 없음';
        final isAbnormal = subtitle.contains('abnormal');
        final selected = selectedIndices.contains(index);
        final rowContent = InkWell(
          onTap: () => _openResult(context, item),
          borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
          child: Padding(
            padding: EdgeInsets.fromLTRB(
              selectionMode ? 0 : MedicalTokens.spaceSm,
              8,
              MedicalTokens.spaceSm,
              8,
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      border: Border.all(color: MedicalTokens.border),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Image.memory(
                      item.originalImageBytes,
                      width: 54,
                      height: 54,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),
                const SizedBox(width: MedicalTokens.spaceSm),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.filename,
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w800,
                            ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        _formatDateTime(item.createdAt),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Theme.of(context).colorScheme.onSurfaceVariant,
                            ),
                      ),
                    ],
                  ),
                ),
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    MedicalBadge(
                      text: subtitle,
                      backgroundColor:
                          isAbnormal ? const Color(0xFFFFEEE8) : MedicalTokens.primarySoft,
                      foregroundColor:
                          isAbnormal ? const Color(0xFFC46235) : MedicalTokens.textMain,
                    ),
                    const SizedBox(height: 6),
                    const Icon(Icons.chevron_right, size: 18),
                  ],
                ),
              ],
            ),
          ),
        );

        return MedicalCard(
          padding: const EdgeInsets.symmetric(
            horizontal: MedicalTokens.spaceXs,
            vertical: MedicalTokens.spaceXs,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              if (selectionMode)
                Checkbox(
                  value: selected,
                  visualDensity: VisualDensity.compact,
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  onChanged: (_) => onToggleSelection(index),
                ),
              Expanded(child: rowContent),
            ],
          ),
        );
      },
    );
  }
}

class _HistoryDashboardView extends StatelessWidget {
  const _HistoryDashboardView({
    required this.entries,
    required this.selectionMode,
    required this.selectedIndices,
    required this.onToggleSelection,
  });

  final List<AnalysisHistoryEntry> entries;
  final bool selectionMode;
  final Set<int> selectedIndices;
  final ValueChanged<int> onToggleSelection;

  /// 카드 하단 메타 바(이름·날짜) 고정 높이 — 그리드 `childAspectRatio` 계산과 맞춘다.
  static const double _metaBarHeight = 68;

  static const Color _galleryImageBg = Color(0xFFF2F5F9);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final crossAxisCount = width >= 1200 ? 4 : (width >= 900 ? 3 : 2);

        final gridHorizontalPadding = MedicalTokens.spaceMd * 2;
        final innerCrossExtent = (width - gridHorizontalPadding).clamp(0.0, double.infinity);
        final crossAxisSpacingTotal = MedicalTokens.spaceSm * (crossAxisCount - 1);
        final cellCrossExtent =
            (innerCrossExtent - crossAxisSpacingTotal) / crossAxisCount;
        final cellMainExtent = cellCrossExtent + _metaBarHeight;
        final childAspectRatio = cellCrossExtent / cellMainExtent;

        return GridView.builder(
          padding: const EdgeInsets.all(MedicalTokens.spaceMd),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: MedicalTokens.spaceSm,
            mainAxisSpacing: MedicalTokens.spaceSm,
            childAspectRatio: childAspectRatio,
          ),
          itemCount: entries.length,
          itemBuilder: (context, index) {
            final item = entries[index];
            final selected =
                selectionMode && selectedIndices.contains(index);

            final card = InkWell(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
              onTap: () => _openResult(context, item),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
                  border: Border.all(
                    color: selected ? Theme.of(context).colorScheme.primary : MedicalTokens.border,
                    width: selected ? 2 : 1,
                  ),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusLg - 1),
                  child: Padding(
                    padding: const EdgeInsets.all(1),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        AspectRatio(
                          aspectRatio: 1,
                          child: ClipRRect(
                            borderRadius: BorderRadius.vertical(
                              top: Radius.circular(MedicalTokens.radiusLg - 2),
                            ),
                            child: ColoredBox(
                              color: _galleryImageBg,
                              child: Image.memory(
                                item.originalImageBytes,
                                fit: BoxFit.contain,
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: _metaBarHeight,
                          child: Container(
                            width: double.infinity,
                            padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
                            decoration: const BoxDecoration(
                              color: Color(0xFFF8FAFD),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: Text(
                                        item.filename,
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                              fontWeight: FontWeight.w800,
                                            ),
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    _HistoryJudgmentIcon(label: item.response.label),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _formatDateTime(item.createdAt),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color:
                                            Theme.of(context).colorScheme.onSurfaceVariant,
                                      ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );

            if (!selectionMode) return card;

            return Stack(
              clipBehavior: Clip.none,
              children: [
                card,
                Positioned(
                  top: 6,
                  left: 6,
                  child: Material(
                    color: Colors.white.withValues(alpha: 0.92),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                      side: BorderSide(color: MedicalTokens.border),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: InkWell(
                      onTap: () => onToggleSelection(index),
                      child: SizedBox(
                        width: 28,
                        height: 28,
                        child: Checkbox(
                          value: selectedIndices.contains(index),
                          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          visualDensity: VisualDensity.compact,
                          onChanged: (_) => onToggleSelection(index),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }
}

class _HistoryJudgmentIcon extends StatelessWidget {
  const _HistoryJudgmentIcon({required this.label});

  final String? label;

  @override
  Widget build(BuildContext context) {
    final raw = (label ?? '').toLowerCase();
    if (raw.contains('abnormal') || raw.contains('이상')) {
      return const Icon(Icons.warning_amber_rounded, size: 18, color: Color(0xFFC46235));
    }
    if (raw.isEmpty || raw.contains('판정 없음')) {
      return Icon(
        Icons.help_outline_rounded,
        size: 18,
        color: Theme.of(context).colorScheme.onSurfaceVariant,
      );
    }
    return const Icon(Icons.check_circle_rounded, size: 18, color: MedicalTokens.success);
  }
}

void _openResult(BuildContext context, AnalysisHistoryEntry item) {
  Navigator.pushNamed(
    context,
    '/result',
    arguments: ResultScreenArgs(
      originalImageBytes: item.originalImageBytes,
      analyzeResponse: item.response,
    ),
  );
}

String _formatDateTime(DateTime value) {
  final y = value.year.toString().padLeft(4, '0');
  final m = value.month.toString().padLeft(2, '0');
  final d = value.day.toString().padLeft(2, '0');
  final hh = value.hour.toString().padLeft(2, '0');
  final mm = value.minute.toString().padLeft(2, '0');
  return '$y-$m-$d $hh:$mm';
}
