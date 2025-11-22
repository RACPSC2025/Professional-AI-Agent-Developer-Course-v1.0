# Parte 1: Document Loaders - La Puerta de Entrada al RAG

![Document Loaders](https://img.shields.io/badge/RAG_Pipeline-Document_Loaders-4A90E2?style=for-the-badge)

## 📖 Índice
1. [Fundamentos Conceptuales](#fundamentos-conceptuales)
2. [Tipos de Documentos y Desafíos](#tipos-de-documentos-y-desafíos)
3. [Document Loaders en LangChain](#document-loaders-en-langchain)
4. [Implementación Paso a Paso](#implementación-paso-a-paso)
5. [Mejores Prácticas](#mejores-prácticas)

---

## 🎯 Fundamentos Conceptuales

### ¿Qué es un Document Loader?

Un **Document Loader** es el primer componente crítico en cualquier pipeline RAG. Su responsabilidad es transformar datos en bruto (PDFs, páginas web, bases de datos, etc.) en un formato estructurado que el sistema pueda procesar.

```mermaid
graph LR
    A[Fuentes de Datos] --> B[Document Loader]
    B --> C[Documentos Estructurados]
    C --> D[Text Splitter]
    D --> E[Embeddings]
    E --> F[Vector Store]
    
    style B fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
```

### ¿Por Qué Son Importantes?

Los Document Loaders no solo leen archivos, sino que:

1. **Preservan Metadata**: Información crucial como fuente, fecha, autor, página
2. **Normalizan Formatos**: Convierten diferentes formatos a una estructura común
3. **Manejan Errores**: Procesan archivos corruptos o mal formateados
4. **Optimizan Performance**: Cargan datos de manera eficiente (lazy loading, streaming)

> [!IMPORTANT]
> **La calidad de tu RAG nunca será mejor que la calidad de tu ingesta de datos**. Un Document Loader mal configurado puede:
> - Perder información crítica (tablas, imágenes, formato)
> - Introducir ruido (headers, footers, elementos de navegación)
> - Fallar silenciosamente (errores no manejados)

---

## 📚 Tipos de Documentos y Desafíos

### Clasificación de Documentos

| Tipo | Ejemplos | Desafíos Principales |
|------|----------|---------------------|
| **Texto Plano** | `.txt`, `.md`, `.csv` | Encoding, delimitadores |
| **Documentos Estructurados** | `.pdf`, `.docx`, `.pptx` | Extracción de layout, tablas, imágenes |
| **Web** | HTML, APIs | JavaScript dinámico, rate limiting |
| **Bases de Datos** | SQL, NoSQL | Esquemas complejos, relaciones |
| **Código** | `.py`, `.js`, `.java` | Sintaxis, dependencias |
| **Multimedia** | Imágenes, Audio, Video | Transcripción, OCR, multimodalidad |

### Desafíos Comunes

#### 1. **Documentos No Estructurados**
```python
# ❌ Problema: PDF con layout complejo
# Texto extraído: "Columna1Texto Columna2Texto TablaHeader"
# Resultado: Contexto mezclado e inútil
```

#### 2. **Metadata Faltante**
```python
# ❌ Sin metadata
doc = Document(page_content="Python es un lenguaje...")

# ✅ Con metadata rica
doc = Document(
    page_content="Python es un lenguaje...",
    metadata={
        "source": "python_guide.pdf",
        "page": 5,
        "author": "Guido van Rossum",
        "date": "2024-01-15",
        "section": "Introducción",
        "language": "es"
    }
)
```

#### 3. **Encoding y Caracteres Especiales**
```python
# ❌ Error común
with open("documento.txt") as f:  # Asume UTF-8
    text = f.read()  # UnicodeDecodeError con Latin-1

# ✅ Manejo robusto
import chardet

with open("documento.txt", "rb") as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]
    text = raw_data.decode(encoding)
```

---

## 🔧 Document Loaders en LangChain

### Arquitectura de Document Loaders

LangChain proporciona una interfaz unificada para cargar documentos:

```python
from langchain.schema import Document

# Estructura base de un Document
class Document:
    page_content: str      # El texto del documento
    metadata: dict         # Información adicional
```

### Loaders Principales

#### 1. **TextLoader** - Archivos de Texto Plano

```python
from langchain_community.document_loaders import TextLoader

# Uso básico
loader = TextLoader("documento.txt", encoding="utf-8")
documents = loader.load()

print(f"Documentos cargados: {len(documents)}")
print(f"Contenido: {documents[0].page_content[:100]}...")
print(f"Metadata: {documents[0].metadata}")
```

**Cuándo usar**: Archivos `.txt`, `.md`, `.log`, código fuente

#### 2. **PyPDFLoader** - Documentos PDF

```python
from langchain_community.document_loaders import PyPDFLoader

# Carga PDF con metadata por página
loader = PyPDFLoader("manual_tecnico.pdf")
pages = loader.load()

# Cada página es un Document separado
for i, page in enumerate(pages):
    print(f"Página {i+1}:")
    print(f"  Contenido: {page.page_content[:100]}...")
    print(f"  Metadata: {page.metadata}")
```

**Características**:
- ✅ Extrae texto página por página
- ✅ Preserva número de página en metadata
- ❌ No extrae imágenes ni tablas complejas

#### 3. **UnstructuredPDFLoader** - PDFs Complejos

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# Para PDFs con layout complejo, tablas, imágenes
loader = UnstructuredPDFLoader(
    "informe_complejo.pdf",
    mode="elements"  # "single" | "elements"
)
documents = loader.load()

# mode="elements" separa por tipo de elemento
for doc in documents:
    element_type = doc.metadata.get("category", "unknown")
    print(f"Tipo: {element_type}")
    print(f"Contenido: {doc.page_content[:100]}...")
```

**Ventajas**:
- ✅ Detecta tablas, títulos, listas
- ✅ Preserva estructura del documento
- ⚠️ Requiere dependencias adicionales (`unstructured`, `pdf2image`)

#### 4. **WebBaseLoader** - Páginas Web

```python
from langchain_community.document_loaders import WebBaseLoader

# Cargar contenido de una URL
loader = WebBaseLoader("https://python.langchain.com/docs/")
documents = loader.load()

print(f"Título: {documents[0].metadata.get('title')}")
print(f"URL: {documents[0].metadata.get('source')}")
print(f"Contenido: {documents[0].page_content[:200]}...")
```

**Características**:
- ✅ Extrae texto limpio (sin HTML)
- ✅ Maneja JavaScript básico
- ❌ No ejecuta JavaScript complejo (usar Playwright para eso)

#### 5. **DirectoryLoader** - Múltiples Archivos

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Cargar todos los archivos .md de un directorio
loader = DirectoryLoader(
    "docs/",
    glob="**/*.md",           # Patrón de archivos
    loader_cls=TextLoader,    # Loader a usar
    show_progress=True,       # Barra de progreso
    use_multithreading=True   # Procesamiento paralelo
)

documents = loader.load()
print(f"Total documentos: {len(documents)}")
```

**Uso profesional**: Ingestar repositorios de documentación completos

#### 6. **CSVLoader** - Datos Tabulares

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Cada fila se convierte en un Document
loader = CSVLoader(
    file_path="productos.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["id", "nombre", "descripcion", "precio"]
    }
)

documents = loader.load()

# Metadata incluye número de fila
for doc in documents[:3]:
    print(f"Fila {doc.metadata['row']}: {doc.page_content}")
```

---

## 💻 Implementación Paso a Paso

### Ejemplo 1: Loader Básico con Manejo de Errores

```python
"""
Ejemplo Básico: Document Loader Robusto
Objetivo: Cargar documentos PDF con manejo de errores y metadata enriquecida
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from pathlib import Path
from typing import List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustPDFLoader:
    """Loader de PDFs con manejo de errores y metadata enriquecida"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        
    def load(self) -> List[Document]:
        """Carga el PDF con validación y enriquecimiento de metadata"""
        
        # Validar que el archivo existe
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")
        
        # Validar extensión
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Archivo no es PDF: {self.file_path}")
        
        try:
            # Cargar PDF
            loader = PyPDFLoader(str(self.file_path))
            documents = loader.load()
            
            # Enriquecer metadata
            for doc in documents:
                doc.metadata.update({
                    "filename": self.file_path.name,
                    "file_size_kb": self.file_path.stat().st_size / 1024,
                    "file_extension": self.file_path.suffix,
                    "total_pages": len(documents)
                })
            
            logger.info(f"✅ Cargado: {self.file_path.name} ({len(documents)} páginas)")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error cargando {self.file_path.name}: {str(e)}")
            raise


# Uso
if __name__ == "__main__":
    loader = RobustPDFLoader("manual_usuario.pdf")
    docs = loader.load()
    
    # Inspeccionar primer documento
    print(f"\n📄 Documento 1:")
    print(f"Contenido (primeros 200 chars): {docs[0].page_content[:200]}...")
    print(f"\nMetadata:")
    for key, value in docs[0].metadata.items():
        print(f"  {key}: {value}")
```

### Ejemplo 2: Loader Multi-Formato con Factory Pattern

```python
"""
Ejemplo Intermedio: Loader Multi-Formato
Objetivo: Sistema que detecta automáticamente el tipo de archivo y usa el loader apropiado
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from pathlib import Path
from typing import List, Type
from abc import ABC, abstractmethod


class DocumentLoaderFactory:
    """Factory para crear loaders según el tipo de archivo"""
    
    # Mapeo de extensiones a loaders
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: str):
        """Crea el loader apropiado según la extensión del archivo"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        loader_class = cls.LOADER_MAP.get(extension)
        if not loader_class:
            raise ValueError(
                f"Tipo de archivo no soportado: {extension}\n"
                f"Soportados: {list(cls.LOADER_MAP.keys())}"
            )
        
        return loader_class(str(path))
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """Carga un documento usando el loader apropiado"""
        loader = cls.create_loader(file_path)
        return loader.load()


class BatchDocumentLoader:
    """Carga múltiples documentos de diferentes formatos"""
    
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        
    def load_all(self) -> List[Document]:
        """Carga todos los documentos"""
        all_documents = []
        
        for file_path in self.file_paths:
            try:
                docs = DocumentLoaderFactory.load_document(file_path)
                all_documents.extend(docs)
                print(f"✅ Cargado: {Path(file_path).name} ({len(docs)} docs)")
            except Exception as e:
                print(f"❌ Error con {Path(file_path).name}: {str(e)}")
                continue
        
        return all_documents


# Uso
if __name__ == "__main__":
    # Lista de archivos de diferentes formatos
    files = [
        "documentos/manual.pdf",
        "documentos/readme.md",
        "documentos/datos.csv",
        "documentos/informe.docx"
    ]
    
    # Cargar todos
    batch_loader = BatchDocumentLoader(files)
    all_docs = batch_loader.load_all()
    
    print(f"\n📊 Total documentos cargados: {len(all_docs)}")
    
    # Agrupar por tipo
    by_type = {}
    for doc in all_docs:
        ext = doc.metadata.get("source", "").split(".")[-1]
        by_type[ext] = by_type.get(ext, 0) + 1
    
    print("\n📈 Distribución por tipo:")
    for ext, count in by_type.items():
        print(f"  .{ext}: {count} documentos")
```

### Ejemplo 3: Loader Personalizado para Formato Propietario

```python
"""
Ejemplo Avanzado: Custom Document Loader
Objetivo: Crear un loader personalizado para un formato JSON específico
"""

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List, Iterator
import json
from pathlib import Path


class CustomJSONLoader(BaseLoader):
    """
    Loader personalizado para archivos JSON con estructura específica.
    
    Formato esperado:
    {
        "articles": [
            {
                "id": "123",
                "title": "Título",
                "content": "Contenido...",
                "author": "Autor",
                "date": "2024-01-15",
                "tags": ["tag1", "tag2"]
            }
        ]
    }
    """
    
    def __init__(
        self,
        file_path: str,
        content_key: str = "content",
        metadata_keys: List[str] = None
    ):
        self.file_path = Path(file_path)
        self.content_key = content_key
        self.metadata_keys = metadata_keys or ["title", "author", "date", "tags"]
        
    def load(self) -> List[Document]:
        """Carga todos los documentos"""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Carga lazy (generador) para archivos grandes.
        Ventaja: No carga todo en memoria de una vez.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validar estructura
        if "articles" not in data:
            raise ValueError("JSON debe contener clave 'articles'")
        
        # Procesar cada artículo
        for article in data["articles"]:
            # Extraer contenido
            content = article.get(self.content_key, "")
            
            if not content:
                continue  # Skip artículos sin contenido
            
            # Construir metadata
            metadata = {
                "source": str(self.file_path),
                "format": "custom_json"
            }
            
            for key in self.metadata_keys:
                if key in article:
                    metadata[key] = article[key]
            
            # Crear documento
            yield Document(
                page_content=content,
                metadata=metadata
            )


class EnhancedJSONLoader(CustomJSONLoader):
    """Versión mejorada con validación y transformación"""
    
    def __init__(
        self,
        file_path: str,
        content_key: str = "content",
        metadata_keys: List[str] = None,
        min_content_length: int = 50,
        transform_content: bool = True
    ):
        super().__init__(file_path, content_key, metadata_keys)
        self.min_content_length = min_content_length
        self.transform_content = transform_content
    
    def _clean_content(self, content: str) -> str:
        """Limpia y normaliza el contenido"""
        # Eliminar espacios múltiples
        content = " ".join(content.split())
        
        # Eliminar caracteres de control
        content = "".join(char for char in content if char.isprintable() or char in "\n\t")
        
        return content.strip()
    
    def lazy_load(self) -> Iterator[Document]:
        """Carga con validación y transformación"""
        for doc in super().lazy_load():
            # Validar longitud mínima
            if len(doc.page_content) < self.min_content_length:
                continue
            
            # Transformar contenido si está habilitado
            if self.transform_content:
                doc.page_content = self._clean_content(doc.page_content)
            
            # Añadir metadata adicional
            doc.metadata["content_length"] = len(doc.page_content)
            doc.metadata["word_count"] = len(doc.page_content.split())
            
            yield doc


# Uso
if __name__ == "__main__":
    # Crear archivo JSON de ejemplo
    sample_data = {
        "articles": [
            {
                "id": "1",
                "title": "Introducción a RAG",
                "content": "RAG (Retrieval-Augmented Generation) es una técnica que combina recuperación de información con generación de lenguaje natural...",
                "author": "Juan Pérez",
                "date": "2024-01-15",
                "tags": ["RAG", "NLP", "AI"]
            },
            {
                "id": "2",
                "title": "Document Loaders",
                "content": "Los document loaders son componentes esenciales en cualquier pipeline RAG...",
                "author": "María García",
                "date": "2024-01-20",
                "tags": ["RAG", "LangChain"]
            }
        ]
    }
    
    # Guardar archivo de ejemplo
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    # Cargar con loader personalizado
    loader = EnhancedJSONLoader(
        "articles.json",
        min_content_length=50,
        transform_content=True
    )
    
    documents = loader.load()
    
    print(f"📚 Documentos cargados: {len(documents)}\n")
    
    for i, doc in enumerate(documents, 1):
        print(f"Documento {i}:")
        print(f"  Título: {doc.metadata.get('title')}")
        print(f"  Autor: {doc.metadata.get('author')}")
        print(f"  Palabras: {doc.metadata.get('word_count')}")
        print(f"  Tags: {doc.metadata.get('tags')}")
        print(f"  Contenido: {doc.page_content[:100]}...\n")
```

---

## ✅ Mejores Prácticas

### 1. **Siempre Enriquecer Metadata**

```python
# ❌ Metadata mínima
doc = Document(page_content=text)

# ✅ Metadata rica
doc = Document(
    page_content=text,
    metadata={
        "source": "documento.pdf",
        "page": 5,
        "section": "Capítulo 3",
        "author": "Juan Pérez",
        "date": "2024-01-15",
        "language": "es",
        "doc_type": "technical_manual",
        "version": "2.0"
    }
)
```

**Por qué**: La metadata permite filtrado preciso durante retrieval.

### 2. **Manejo Robusto de Errores**

```python
def load_documents_safely(file_paths: List[str]) -> List[Document]:
    """Carga documentos con manejo de errores"""
    documents = []
    errors = []
    
    for path in file_paths:
        try:
            loader = DocumentLoaderFactory.create_loader(path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            errors.append({"file": path, "error": str(e)})
            logger.error(f"Error cargando {path}: {e}")
    
    # Reportar errores al final
    if errors:
        logger.warning(f"⚠️ {len(errors)} archivos fallaron")
        for error in errors:
            logger.warning(f"  - {error['file']}: {error['error']}")
    
    return documents
```

### 3. **Lazy Loading para Archivos Grandes**

```python
# ❌ Carga todo en memoria
documents = loader.load()  # Puede causar OOM con archivos grandes

# ✅ Lazy loading (generador)
for document in loader.lazy_load():
    process_document(document)  # Procesa uno a la vez
```

### 4. **Validación de Contenido**

```python
def validate_document(doc: Document) -> bool:
    """Valida que un documento sea útil"""
    
    # Contenido mínimo
    if len(doc.page_content) < 50:
        return False
    
    # No solo espacios en blanco
    if not doc.page_content.strip():
        return False
    
    # Metadata esencial presente
    required_metadata = ["source"]
    if not all(key in doc.metadata for key in required_metadata):
        return False
    
    return True

# Filtrar documentos inválidos
valid_docs = [doc for doc in documents if validate_document(doc)]
```

### 5. **Logging y Observabilidad**

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_with_logging(file_path: str) -> List[Document]:
    """Carga documentos con logging detallado"""
    start_time = datetime.now()
    
    logger.info(f"🔄 Iniciando carga: {file_path}")
    
    try:
        loader = DocumentLoaderFactory.create_loader(file_path)
        documents = loader.load()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"✅ Carga exitosa: {file_path}\n"
            f"   Documentos: {len(documents)}\n"
            f"   Duración: {duration:.2f}s"
        )
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error cargando {file_path}: {str(e)}")
        raise
```

---

## 🎯 Resumen y Próximos Pasos

### Lo que Aprendimos

✅ **Document Loaders** son el primer paso crítico en RAG  
✅ **Metadata rica** mejora significativamente la calidad del retrieval  
✅ **Manejo de errores** es esencial para sistemas en producción  
✅ **Diferentes formatos** requieren diferentes loaders  
✅ **Lazy loading** optimiza memoria para archivos grandes  

### Checklist de Implementación

- [ ] Identificar todos los formatos de documentos en tu sistema
- [ ] Seleccionar loaders apropiados para cada formato
- [ ] Implementar enriquecimiento de metadata
- [ ] Añadir manejo robusto de errores
- [ ] Configurar logging y monitoreo
- [ ] Validar calidad de documentos cargados

### Próximo Paso

Una vez que tus documentos están cargados, el siguiente paso es **dividirlos en chunks** para optimizar el retrieval.

➡️ **[Continuar a Parte 2: Text Splitters](02_text_splitters.md)**

---

<div align="center">

**[⬅️ Volver al Módulo 5](README.md)** | **[Siguiente: Text Splitters ➡️](02_text_splitters.md)**

</div>
