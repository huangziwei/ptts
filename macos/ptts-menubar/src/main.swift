import Cocoa

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var toggleItem: NSMenuItem!
    private var process: Process?
    private var logHandle: FileHandle?
    private var startupTimer: DispatchSourceTimer?
    private var startupAttempts = 0
    private var isChecking = false
    private var isStarting = false
    private var activeRepoPath: String?
    private let maxStartupAttempts = 40
    private let startupInterval: TimeInterval = 0.5

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
        maybePromptInstallSymlink()
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
        guard let repoPath = resolveRepoPath() else {
            showAlert(
                title: "pTTS folder not found",
                message: "Place the app inside the pTTS repo (e.g. macos/ptts-menubar/build), or set PTTS_ROOT to the repo root."
            )
            return
        }

        activeRepoPath = repoPath
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
            isStarting = true
            updateMenuState()
            beginStartupChecks()
        } catch {
            cleanupProcessState()
            showAlert(title: "Failed to start pTTS", message: error.localizedDescription)
        }
    }

    private func stopServer() {
        if let process = process, process.isRunning {
            process.terminate()
        }
        #if !arch(arm64)
        stopPodmanServerAsync()
        #endif
        cleanupProcessState()
    }

    private func cleanupProcessState() {
        process = nil
        cancelStartupChecks()
        if let handle = logHandle {
            try? handle.close()
        }
        logHandle = nil
        isStarting = false
        activeRepoPath = nil
        updateMenuState()
    }

    private func updateMenuState() {
        let running = process?.isRunning == true
        if running && isStarting {
            toggleItem?.title = "Starting..."
            statusItem?.button?.title = "pTTS (starting)"
        } else {
            toggleItem?.title = running ? "Stop Player Server" : "Start Player Server"
            statusItem?.button?.title = running ? "pTTS (on)" : "pTTS"
        }
    }

    private func stopPodmanServerAsync() {
        guard let repoPath = activeRepoPath ?? resolveRepoPath() else { return }
        let env = makeEnvironment()
        guard let podmanPath = resolveExecutable("podman", env: env) else { return }
        let container = pmxContainerName(repoPath: repoPath, env: env)
        let script = """
import os
import signal

killed = 0
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmd = handle.read().decode(errors="ignore").replace("\\x00", " ")
    except (FileNotFoundError, PermissionError):
        continue
    if "ptts" in cmd and "play" in cmd:
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
print(killed)
"""

        DispatchQueue.global(qos: .utility).async { [weak self] in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: podmanPath)
            process.arguments = ["exec", container, "python", "-c", script]
            process.environment = env
            do {
                try process.run()
                process.waitUntilExit()
            } catch {
                DispatchQueue.main.async {
                    self?.showAlert(title: "Stop failed", message: error.localizedDescription)
                }
            }
        }
    }

    private func pmxContainerName(repoPath: String, env: [String: String]) -> String {
        if let name = env["PMX_NAME"], !name.isEmpty {
            return name
        }
        let namespace = env["PMX_NAMESPACE"].flatMap { $0.isEmpty ? nil : $0 } ?? "ptts"
        let baseName = URL(fileURLWithPath: repoPath).lastPathComponent
        let normalized = normalizeContainerBase(baseName)
        return "\(namespace)-\(normalized)"
    }

    private func normalizeContainerBase(_ value: String) -> String {
        var result = ""
        var lastWasDash = false
        for scalar in value.unicodeScalars {
            let v = scalar.value
            let allowed = (v >= 48 && v <= 57)
                || (v >= 65 && v <= 90)
                || (v >= 97 && v <= 122)
                || v == 95
                || v == 46
                || v == 45
            if allowed {
                result.unicodeScalars.append(scalar)
                lastWasDash = false
            } else if !lastWasDash {
                result.append("-")
                lastWasDash = true
            }
        }
        let trimmed = result.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "repo" : trimmed
    }

    private func beginStartupChecks() {
        cancelStartupChecks()
        startupAttempts = 0
        isChecking = false
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + startupInterval, repeating: startupInterval)
        timer.setEventHandler { [weak self] in
            self?.checkServerReady()
        }
        startupTimer = timer
        timer.resume()
    }

    private func cancelStartupChecks() {
        startupTimer?.cancel()
        startupTimer = nil
        isChecking = false
    }

    private func checkServerReady() {
        guard process?.isRunning == true else {
            cancelStartupChecks()
            DispatchQueue.main.async { [weak self] in
                self?.isStarting = false
                self?.updateMenuState()
            }
            return
        }

        if startupAttempts >= maxStartupAttempts {
            cancelStartupChecks()
            DispatchQueue.main.async { [weak self] in
                self?.isStarting = false
                self?.updateMenuState()
                self?.showAlert(
                    title: "Server taking too long",
                    message: "pTTS did not respond in time. Check the log for details."
                )
            }
            return
        }

        if isChecking {
            return
        }

        startupAttempts += 1
        isChecking = true

        guard let url = playerURL() else {
            isChecking = false
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 0.8
        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            guard let self else { return }
            self.isChecking = false
            if error == nil, response is HTTPURLResponse {
                self.cancelStartupChecks()
                DispatchQueue.main.async {
                    self.isStarting = false
                    self.updateMenuState()
                    self.openPlayerURL()
                }
            }
        }.resume()
    }

    private func maybePromptInstallSymlink() {
        guard shouldPromptInstall() else { return }

        let alert = NSAlert()
        alert.messageText = "Add to Spotlight?"
        alert.informativeText = "Create a symlink in /Applications so you can open pTTS Menubar from Spotlight?"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Create Symlink")
        alert.addButton(withTitle: "Not Now")

        let response = alert.runModal()
        defer { markInstallPrompted() }

        guard response == .alertFirstButtonReturn else { return }
        do {
            try installSymlink()
        } catch {
            showAlert(title: "Symlink failed", message: error.localizedDescription)
        }
    }

    private func shouldPromptInstall() -> Bool {
        if hasPromptedInstall() {
            return false
        }

        let systemApps = URL(fileURLWithPath: "/Applications", isDirectory: true)
            .standardizedFileURL
        let appURL = Bundle.main.bundleURL.standardizedFileURL
        return !appURL.path.hasPrefix(systemApps.path + "/")
    }

    private func hasPromptedInstall() -> Bool {
        return FileManager.default.fileExists(atPath: promptMarkerURL().path)
    }

    private func markInstallPrompted() {
        let url = promptMarkerURL()
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
    }

    private func promptMarkerURL() -> URL {
        let supportDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return supportDir
            .appendingPathComponent("ptts-menubar", isDirectory: true)
            .appendingPathComponent("install-prompted")
    }

    private func installSymlink() throws {
        let fm = FileManager.default
        let appsDir = URL(fileURLWithPath: "/Applications", isDirectory: true)

        let appURL = Bundle.main.bundleURL.standardizedFileURL
        let linkURL = appsDir.appendingPathComponent(appURL.lastPathComponent)

        if fm.fileExists(atPath: linkURL.path) {
            if let destinationPath = try? fm.destinationOfSymbolicLink(atPath: linkURL.path) {
                let resolved = URL(fileURLWithPath: destinationPath).standardizedFileURL
                if resolved == appURL {
                    return
                }
            }
            throw NSError(domain: "ptts-menubar", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "An item already exists at \(linkURL.path). Remove it first if you want to create a symlink."
            ])
        }

        try fm.createSymbolicLink(at: linkURL, withDestinationURL: appURL)
    }

    private func resolveRepoPath() -> String? {
        if let override = ProcessInfo.processInfo.environment["PTTS_ROOT"], !override.isEmpty {
            if FileManager.default.fileExists(atPath: override) {
                return override
            }
        }

        var current = Bundle.main.bundleURL
        for _ in 0..<8 {
            if isRepoRoot(current) {
                return current.path
            }
            current.deleteLastPathComponent()
        }

        return nil
    }

    private func isRepoRoot(_ url: URL) -> Bool {
        let fm = FileManager.default
        let pyproject = url.appendingPathComponent("pyproject.toml").path
        let binDir = url.appendingPathComponent("bin").path
        let pttsDir = url.appendingPathComponent("ptts").path
        return fm.fileExists(atPath: pyproject)
            && fm.fileExists(atPath: binDir)
            && fm.fileExists(atPath: pttsDir)
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

    private func playerURL() -> URL? {
        let urlString = ProcessInfo.processInfo.environment["PTTS_PLAYER_URL"] ?? "http://localhost:1912"
        return URL(string: urlString)
    }

    private func openPlayerURL() {
        guard let url = playerURL() else { return }
        NSWorkspace.shared.open(url)
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
