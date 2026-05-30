import 'dart:async';

import 'package:flutter/material.dart';
import 'package:http/http.dart' show ClientException;

import '../api/eye_api_client.dart';
import 'dialog_keyboard.dart';

/// 안내 텍스트 한 줄 + OK (에러 코드 없음).
Future<void> showNoticeDialog(
  BuildContext context, {
  required String message,
}) {
  return showDialog<void>(
    context: context,
    barrierDismissible: true,
    builder: (dialogContext) {
      void close() => Navigator.of(dialogContext).pop();
      final bodyStyle = Theme.of(dialogContext).textTheme.bodyMedium?.copyWith(
            height: 1.5,
          );
      return dialogOkShortcuts(
        onClose: close,
        child: AlertDialog(
          content: SelectableText(message, style: bodyStyle),
          actions: [
            TextButton(
              onPressed: close,
              child: const Text('OK'),
            ),
          ],
        ),
      );
    },
  );
}

/// `Error Code:` (강조) + 메시지(다음 줄).
Future<void> showCodeNoticeDialog(
  BuildContext context, {
  required String code,
  required String message,
}) {
  return showDialog<void>(
    context: context,
    barrierDismissible: true,
    builder: (dialogContext) {
      void close() => Navigator.of(dialogContext).pop();
      final theme = Theme.of(dialogContext);
      final codeStyle = theme.textTheme.titleLarge?.copyWith(
        fontWeight: FontWeight.w700,
        height: 1.2,
      );
      final bodyStyle = theme.textTheme.bodyMedium?.copyWith(
        height: 1.5,
      );
      return dialogOkShortcuts(
        onClose: close,
        child: AlertDialog(
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SelectableText('Error Code: $code', style: codeStyle),
              const SizedBox(height: 10),
              SelectableText(message, style: bodyStyle),
            ],
          ),
          actions: [
            TextButton(
              onPressed: close,
              child: const Text('OK'),
            ),
          ],
        ),
      );
    },
  );
}

String? _errorCodeFor(Object error) {
  if (error is EyeApiException) return '${error.statusCode}';
  if (error is TimeoutException) return '408';
  if (error is ClientException) return '503';
  return null;
}

String _errorMessageFor(Object error) {
  if (error is EyeApiException) return parseApiErrorMessage(error.body);
  if (error is TimeoutException) {
    return error.message ?? '요청 시간이 초과되었습니다.';
  }
  if (error is ClientException) {
    final msg = error.message.trim();
    return msg.isEmpty ? '서버에 연결할 수 없습니다.' : msg;
  }
  return error.toString();
}

/// API·네트워크 등 코드가 있는 오류는 `Error Code:` 형식.
Future<void> showErrorNotice(BuildContext context, Object error) {
  final code = _errorCodeFor(error);
  if (code != null) {
    return showCodeNoticeDialog(
      context,
      code: code,
      message: _errorMessageFor(error),
    );
  }
  return showNoticeDialog(context, message: _errorMessageFor(error));
}
