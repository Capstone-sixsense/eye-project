import Cocoa
import FlutterMacOS

/// .app Resources 에 번들된 FastAPI 백엔드를 기동하고 /health 를 대기합니다.
/// 개발 모드(`flutter run`)에서는 Resources 가 없으면 외부 백엔드(8000)를 가정하고 건너뜁니다.
final class BackendLauncher {
  static let shared = BackendLauncher()

  private var process: Process?
  private var logHandle: FileHandle?

  private init() {}

  func bundledBackendAvailable() -> Bool {
    guard let resources = Bundle.main.resourceURL else { return false }
    let script = resources.appendingPathComponent("run_backend.sh")
    let venv = resources.appendingPathComponent("python-venv/bin/uvicorn")
    return FileManager.default.fileExists(atPath: script.path)
      && FileManager.default.fileExists(atPath: venv.path)
  }

  func startAndWait(timeout: TimeInterval = 180) -> Bool {
    guard bundledBackendAvailable() else {
      NSLog("[BackendLauncher] Bundled backend not found — assuming dev server on :8000")
      return waitForHealth(timeout: min(timeout, 5), required: false)
    }

    guard let resources = Bundle.main.resourceURL else { return false }
    let script = resources.appendingPathComponent("run_backend.sh")

    let dataDir = FileManager.default.urls(
      for: .applicationSupportDirectory,
      in: .userDomainMask
    ).first!.appendingPathComponent("EyeProject", isDirectory: true)

    do {
      try FileManager.default.createDirectory(
        at: dataDir.appendingPathComponent("logs"),
        withIntermediateDirectories: true
      )
    } catch {
      NSLog("[BackendLauncher] Failed to create log dir: \(error)")
    }

    let logURL = dataDir.appendingPathComponent("logs/launcher.log")
    FileManager.default.createFile(atPath: logURL.path, contents: nil)
    logHandle = try? FileHandle(forWritingTo: logURL)
    logHandle?.seekToEndOfFile()

    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/bin/bash")
    proc.arguments = [script.path]
    proc.currentDirectoryURL = resources
    var env = ProcessInfo.processInfo.environment
    env["EYE_PROJECT_DATA_DIR"] = dataDir.path
    proc.environment = env
    if let logHandle {
      proc.standardOutput = logHandle
      proc.standardError = logHandle
    }

    do {
      try proc.run()
      process = proc
      NSLog("[BackendLauncher] Started backend pid=\(proc.processIdentifier)")
    } catch {
      NSLog("[BackendLauncher] Failed to start: \(error)")
      return false
    }

    return waitForHealth(timeout: timeout, required: true)
  }

  func stop() {
    guard let process else { return }
    if process.isRunning {
      process.terminate()
      usleep(500_000)
      if process.isRunning {
        process.interrupt()
      }
    }
    self.process = nil
    logHandle?.closeFile()
    logHandle = nil
  }

  private func waitForHealth(timeout: TimeInterval, required: Bool) -> Bool {
    guard let url = URL(string: "http://127.0.0.1:8000/health") else { return false }
    let deadline = Date().addingTimeInterval(timeout)

    while Date() < deadline {
      var request = URLRequest(url: url)
      request.timeoutInterval = 2
      let sem = DispatchSemaphore(value: 0)
      var ok = false
      URLSession.shared.dataTask(with: request) { _, response, _ in
        if let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) {
          ok = true
        }
        sem.signal()
      }.resume()
      _ = sem.wait(timeout: .now() + 3)
      if ok {
        NSLog("[BackendLauncher] /health OK")
        return true
      }
      usleep(500_000)
    }

    if required {
      NSLog("[BackendLauncher] /health timeout after \(timeout)s")
    }
    return !required
  }
}

@main
class AppDelegate: FlutterAppDelegate {
  override func applicationWillFinishLaunching(_ notification: Notification) {
    if !BackendLauncher.shared.startAndWait(timeout: 180) {
      let alert = NSAlert()
      alert.messageText = "백엔드 서버를 시작하지 못했습니다"
      alert.informativeText =
        "분석 API(127.0.0.1:8000)가 준비되지 않았습니다.\n"
        + "~/Library/Application Support/EyeProject/logs/backend.log 를 확인하세요."
      alert.alertStyle = .critical
      alert.runModal()
      NSApp.terminate(nil)
    }
    super.applicationWillFinishLaunching(notification)
  }

  override func applicationWillTerminate(_ notification: Notification) {
    BackendLauncher.shared.stop()
    super.applicationWillTerminate(notification)
  }

  override func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    return true
  }

  override func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
    return true
  }
}
