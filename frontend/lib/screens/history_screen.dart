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

  void _clearHistory() {
    AnalysisHistoryStore.clear();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final entries = AnalysisHistoryStore.entries;

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
            onPressed: entries.isEmpty ? null : _clearHistory,
            icon: const Icon(Icons.delete_outline),
            tooltip: '이력 비우기',
          ),
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
              ? _HistoryListView(entries: entries)
              : _HistoryDashboardView(entries: entries)),
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
  const _HistoryListView({required this.entries});

  final List<AnalysisHistoryEntry> entries;

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
        return MedicalCard(
          padding: const EdgeInsets.symmetric(
            horizontal: MedicalTokens.spaceSm,
            vertical: MedicalTokens.spaceXs,
          ),
          child: ListTile(
            contentPadding: const EdgeInsets.symmetric(horizontal: 4),
            leading: ClipRRect(
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
            title: Text(
              item.filename,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
            ),
            subtitle: Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(_formatDateTime(item.createdAt)),
            ),
            trailing: Column(
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
            onTap: () => _openResult(context, item),
          ),
        );
      },
    );
  }
}

class _HistoryDashboardView extends StatelessWidget {
  const _HistoryDashboardView({required this.entries});

  final List<AnalysisHistoryEntry> entries;

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
            return InkWell(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
              onTap: () => _openResult(context, item),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
                  border: Border.all(color: MedicalTokens.border),
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
                                        color: Theme.of(context).colorScheme.onSurfaceVariant,
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
