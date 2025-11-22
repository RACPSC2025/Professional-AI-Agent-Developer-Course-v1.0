# AI Investment Analyst (Module 1 Project)

Este es el primer proyecto práctico del curso. Un agente simple pero poderoso que combina datos estructurados (precio de acciones) con datos no estructurados (noticias) para dar una recomendación de inversión.

## 🛠️ Tecnologías Usadas
-   **LangChain:** Para orquestar el agente.
-   **OpenAI GPT-4o:** Como cerebro del agente.
-   **YFinance:** Para obtener precios de acciones en tiempo real.
-   **DuckDuckGo Search:** Para buscar noticias recientes.

## 🚀 Cómo Ejecutar

1.  Asegúrate de haber configurado el entorno (ver `README.md` principal).
2.  Activa el entorno virtual:
    ```powershell
    ..\..\venv\Scripts\Activate.ps1
    ```
3.  Ejecuta el script:
    ```powershell
    python investment_analyst.py
    ```
4.  Introduce un símbolo de acción (ej. `AAPL`, `MSFT`, `TSLA`) cuando se te pida.

## 🧠 Cómo Funciona (Under the Hood)
El script utiliza el patrón **OpenAI Tools Agent**.
1.  El LLM recibe tu petición ("Analiza AAPL").
2.  Decide llamar a `get_stock_price("AAPL")`.
3.  Recibe el precio.
4.  Decide llamar a `get_company_news("Apple Inc latest financial news")`.
5.  Recibe las noticias.
6.  Sintetiza toda la información y genera el reporte final.
