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
    return FilledButton(
      onPressed: onPressed,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (leading != null) ...[
            leading!,
            const SizedBox(width: 8),
          ],
          Text(label),
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
  });

  final String label;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton(
      onPressed: onPressed,
      child: Text(label),
    );
  }
}
