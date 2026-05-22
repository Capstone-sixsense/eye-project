import 'package:flutter/material.dart';

class MedicalTokens {
  MedicalTokens._();

  static const Color primary = Color(0xFF72A9C7);
  static const Color primarySoft = Color(0xFFDDEFF8);
  static const Color success = Color(0xFF3BAA8B);
  static const Color caution = Color(0xFFF2B46B);
  static const Color background = Color(0xFFF5F8FB);
  static const Color surface = Colors.white;
  static const Color border = Color(0xFFD6E2EB);
  static const Color textMain = Color(0xFF1F2D3A);
  static const Color textSubtle = Color(0xFF6B7D8D);

  static const double radiusLg = 20;
  static const double radiusMd = 14;
  static const double spaceXs = 8;
  static const double spaceSm = 12;
  static const double spaceMd = 16;
  static const double spaceLg = 24;

  static const List<BoxShadow> cardShadow = [
    BoxShadow(
      color: Color(0x140C2A4A),
      blurRadius: 16,
      offset: Offset(0, 6),
    ),
  ];
}

class MedicalSectionTitle extends StatelessWidget {
  const MedicalSectionTitle(this.title, {super.key, this.subtitle});

  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w700,
            color: MedicalTokens.textMain,
          ),
        ),
        if (subtitle != null) ...[
          const SizedBox(height: 4),
          Text(
            subtitle!,
            style: theme.textTheme.bodySmall?.copyWith(
              color: MedicalTokens.textSubtle,
            ),
          ),
        ],
      ],
    );
  }
}

/// 제목 + 본문, 왼쪽 강조선·아이콘. 업로드·결과 등 안내 문구에 공통 사용.
class MedicalNoticeBanner extends StatelessWidget {
  const MedicalNoticeBanner({
    super.key,
    this.title,
    required this.body,
    this.icon = Icons.info_outline_rounded,
  });

  final String? title;
  final String body;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final titleStyle = theme.textTheme.titleSmall?.copyWith(
      fontWeight: FontWeight.w600,
      color: MedicalTokens.textMain,
    );
    final bodyStyle = theme.textTheme.bodySmall?.copyWith(
      color: MedicalTokens.textSubtle,
      height: 1.45,
    );

    return DecoratedBox(
      decoration: BoxDecoration(
        color: MedicalTokens.primarySoft.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
        border: Border.all(color: MedicalTokens.border),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
        child: DecoratedBox(
          decoration: const BoxDecoration(
            border: Border(
              left: BorderSide(color: MedicalTokens.primary, width: 4),
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 14, 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(icon, size: 20, color: MedicalTokens.primary),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (title != null && title!.isNotEmpty) ...[
                        Text(title!, style: titleStyle),
                        const SizedBox(height: 4),
                      ],
                      Text(body, style: bodyStyle),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class MedicalCard extends StatelessWidget {
  const MedicalCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(MedicalTokens.spaceMd),
  });

  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: MedicalTokens.surface,
        borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
        border: Border.all(color: MedicalTokens.border),
        boxShadow: MedicalTokens.cardShadow,
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}

class MedicalBadge extends StatelessWidget {
  const MedicalBadge({
    super.key,
    required this.text,
    this.backgroundColor = MedicalTokens.primarySoft,
    this.foregroundColor = MedicalTokens.textMain,
  });

  final String text;
  final Color backgroundColor;
  final Color foregroundColor;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Text(
          text,
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: foregroundColor,
                fontWeight: FontWeight.w700,
              ),
        ),
      ),
    );
  }
}

class MedicalPrimaryButton extends StatelessWidget {
  const MedicalPrimaryButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.leading,
  });

  final String label;
  final VoidCallback? onPressed;
  final Widget? leading;

  @override
  Widget build(BuildContext context) {
    final labelWidget = Text(label);
    return FilledButton(
      onPressed: onPressed,
      child: leading == null
          ? labelWidget
          : Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                leading!,
                const SizedBox(width: 8),
                labelWidget,
              ],
            ),
    );
  }
}

class MedicalSecondaryButton extends StatelessWidget {
  const MedicalSecondaryButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.leading,
  });

  final String label;
  final VoidCallback? onPressed;
  final Widget? leading;

  @override
  Widget build(BuildContext context) {
    final labelWidget = Text(label);
    return OutlinedButton(
      onPressed: onPressed,
      child: leading == null
          ? labelWidget
          : Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                leading!,
                const SizedBox(width: 8),
                labelWidget,
              ],
            ),
    );
  }
}
