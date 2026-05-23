import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class _DialogEscapeIntent extends Intent {
  const _DialogEscapeIntent();
}

class _DialogEnterIntent extends Intent {
  const _DialogEnterIntent();
}

/// OK·확인 단일 액션: Enter / Esc → [onClose].
Widget dialogOkShortcuts({
  required VoidCallback onClose,
  required Widget child,
}) {
  return _DialogShortcuts(
    onEscape: onClose,
    onEnter: onClose,
    child: child,
  );
}

/// 확인/취소: Enter → [onConfirm], Esc → [onCancel]만.
Widget dialogConfirmCancelShortcuts({
  required VoidCallback onConfirm,
  required VoidCallback onCancel,
  required Widget child,
}) {
  return _DialogShortcuts(
    onEscape: onCancel,
    onEnter: onConfirm,
    child: child,
  );
}

class _DialogShortcuts extends StatelessWidget {
  const _DialogShortcuts({
    required this.onEscape,
    this.onEnter,
    required this.child,
  });

  final VoidCallback onEscape;
  final VoidCallback? onEnter;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Shortcuts(
      shortcuts: <ShortcutActivator, Intent>{
        const SingleActivator(LogicalKeyboardKey.escape): const _DialogEscapeIntent(),
        if (onEnter != null) ...{
          const SingleActivator(LogicalKeyboardKey.enter): const _DialogEnterIntent(),
          const SingleActivator(LogicalKeyboardKey.numpadEnter): const _DialogEnterIntent(),
        },
      },
      child: Actions(
        actions: <Type, Action<Intent>>{
          _DialogEscapeIntent: CallbackAction<_DialogEscapeIntent>(
            onInvoke: (_) {
              onEscape();
              return null;
            },
          ),
          _DialogEnterIntent: CallbackAction<_DialogEnterIntent>(
            onInvoke: (_) {
              onEnter?.call();
              return null;
            },
          ),
        },
        child: Focus(autofocus: true, child: child),
      ),
    );
  }
}
