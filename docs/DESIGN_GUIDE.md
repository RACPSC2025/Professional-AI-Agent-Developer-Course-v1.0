# 🎨 Course Design Guide & UX Standards

## 🌟 Philosophy
**"Professional, Modern, Dark Mode First."**
We want students to feel like they are using a premium developer tool, not reading a textbook.

## 🎨 Color Palette (Cyberpunk / Enterprise AI)

| Color Name | Hex | Usage |
| :--- | :--- | :--- |
| **Void Black** | `#0E1117` | Main Background (Streamlit/Web) |
| **Deep Space** | `#1E1E1E` | Card/Panel Background |
| **Neon Purple** | `#8E44AD` | Primary Accent (Headers, Buttons) |
| **Cyber Blue** | `#3498DB` | Secondary Accent (Links, Info) |
| **Success Green** | `#2ECC71` | Success States, "Approved" |
| **Alert Red** | `#E74C3C` | Errors, Warnings |
| **Gold** | `#F1C40F` | Capstone/Premium Features |

## 📝 Typography & Formatting

### Headers
-   **H1:** Title Case, with Emoji. (e.g., `# 🚀 Module 1: Foundations`)
-   **H2:** Section Headers. Use clear, action-oriented language.
-   **H3:** Sub-sections.

### Badges (Shields.io)
Use badges at the top of every README to give instant context.
```markdown
![Level](https://img.shields.io/badge/Level-Intermediate-3498DB?style=for-the-badge&logo=python&logoColor=white)
![Time](https://img.shields.io/badge/Time-2_Hours-A7C7E7?style=for-the-badge&labelColor=2D2D44)
```

### Admonitions (GitHub Alerts)
Use these sparingly but effectively.
> [!NOTE]
> For context or side-notes.

> [!IMPORTANT]
> For critical configuration steps (API Keys, etc.).

> [!WARNING]
> For potential pitfalls or costs.

## 📊 Diagrams (Mermaid)
Always use Mermaid for architecture. It's editable and renders natively.
*   **Flowcharts:** For logic paths.
*   **Sequence Diagrams:** For agent interactions.
*   **Class Diagrams:** For code structure.

## 💻 Code Blocks
-   Always specify the language: \`\`\`python
-   Use comments to explain *why*, not just *what*.
-   Keep snippets copy-pasteable (include imports).

## 🧠 UX Principles for Students
1.  **"Time to Hello World":** The first code snippet should run in < 5 minutes.
2.  **Visual Feedback:** Agents should print emojis or use progress bars.
3.  **Fail Gracefully:** Scripts should check for API Keys and give helpful errors, not stack traces.
