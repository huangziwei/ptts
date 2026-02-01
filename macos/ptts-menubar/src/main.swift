import Cocoa

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let repoPath: String
    private var statusItem: NSStatusItem!
    private var toggleItem: NSMenuItem!
    private var process: Process?
    private var logHandle: FileHandle?

    override init() {
        let defaultRepoPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("projects/ptts")
            .path
        repoPath = ProcessInfo.processInfo.environment["PTTS_ROOT"] ?? defaultRepoPath
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.title = "pTTS"
        }

        let menu = NSMenu()
        toggleItem = NSMenuItem(title: "Start Player Server", action: #selector(toggleServer), keyEquivalent: "")
        toggleItem.target = self
        menu.addItem(toggleItem)
        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
        updateMenuState()
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopServer()
    }

    @objc private func toggleServer() {
        if process?.isRunning == true {
            stopServer()
        } else {
            startServer()
        }
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    private func startServer() {
        guard FileManager.default.fileExists(atPath: repoPath) else {
            showAlert(title: "pTTS folder not found", message: "Set PTTS_ROOT or update the default path.\nCurrent: \(repoPath)")
            return
        }

        let env = makeEnvironment()
        let process = Process()
        process.currentDirectoryURL = URL(fileURLWithPath: repoPath)
        process.environment = env

        #if arch(arm64)
        guard resolveExecutable("uv", env: env) != nil else {
            showAlert(title: "uv not found", message: "Make sure uv is on your PATH.")
            return
        }
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["uv", "run", "ptts", "play"]
        #else
        let pmxPath = URL(fileURLWithPath: repoPath).appendingPathComponent("bin/pmx").path
        guard FileManager.default.isExecutableFile(atPath: pmxPath) else {
            showAlert(title: "pmx not found", message: "Expected executable at: \(pmxPath)")
            return
        }
        guard resolveExecutable("podman", env: env) != nil else {
            showAlert(title: "podman not found", message: "Make sure podman is installed and on your PATH.")
            return
        }
        process.executableURL = URL(fileURLWithPath: pmxPath)
        process.arguments = ["uv", "run", "ptts", "play"]
        #endif

        do {
            let logURL = try ensureLogFile()
            let handle = try FileHandle(forWritingTo: logURL)
            handle.seekToEndOfFile()
            process.standardOutput = handle
            process.standardError = handle
            logHandle = handle

            process.terminationHandler = { [weak self] _ in
                DispatchQueue.main.async {
                    self?.cleanupProcessState()
                }
            }

            try process.run()
            self.process = process
            updateMenuState()
        } catch {
            cleanupProcessState()
            showAlert(title: "Failed to start pTTS", message: error.localizedDescription)
        }
    }

    private func stopServer() {
        if let process = process, process.isRunning {
            process.terminate()
        }
        cleanupProcessState()
    }

    private func cleanupProcessState() {
        process = nil
        if let handle = logHandle {
            try? handle.close()
        }
        logHandle = nil
        updateMenuState()
    }

    private func updateMenuState() {
        let running = process?.isRunning == true
        toggleItem?.title = running ? "Stop Player Server" : "Start Player Server"
        statusItem?.button?.title = running ? "pTTS (on)" : "pTTS"
    }

    private func ensureLogFile() throws -> URL {
        let logDir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Logs", isDirectory: true)
        if !FileManager.default.fileExists(atPath: logDir.path) {
            try FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        }

        let logURL = logDir.appendingPathComponent("ptts-menubar.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }

        return logURL
    }

    private func makeEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let fallbackPaths = [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/opt/podman/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]
        var parts = env["PATH"].map { $0.split(separator: ":").map(String.init) } ?? []
        for path in fallbackPaths where !parts.contains(path) {
            parts.append(path)
        }
        env["PATH"] = parts.joined(separator: ":")
        return env
    }

    private func resolveExecutable(_ name: String, env: [String: String]) -> String? {
        guard let pathValue = env["PATH"] else { return nil }
        for dir in pathValue.split(separator: ":") {
            let candidate = String(dir) + "/" + name
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.runModal()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
