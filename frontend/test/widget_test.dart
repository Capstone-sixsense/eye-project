import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:eye_project/screens/upload_screen.dart';

void main() {
  testWidgets('upload screen builds', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: UploadScreen(),
      ),
    );
    expect(find.text('Upload Retinal Image'), findsOneWidget);
  });
}
