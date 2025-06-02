import google.generativeai as genai


class GeminiClient:

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro-preview-05-06"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self._configure_model()

    def __call__(self, message: str) -> str:
        if not self.model:
            raise RuntimeError("Model is not initialized.")
        response = self.model.generate_content(message)
        return response.text

    def _configure_model(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)


class Chunk:

    def __init__(self, text: str):
        self.text = text


class LLMChunker:

    prompt = """
I will provide you with an article, where each line is prefixed with a line number (starting from 0). Your task is to segment this article into semantically meaningful paragraphs.

If the article already contains clearly defined sections or chapters, you should further divide each section into multiple paragraphs, based on their semantic structure.

Please strictly output only the line number ranges for each paragraph, using the format:

start_line end_line
start_line end_line
...

Make sure:

The first paragraph starts at line 0.

The last paragraph ends at the final line of the article.

There are no missing or overlapping lines between paragraphs.

Do not include any other output or explanation — only the list of line number ranges.

The article is given below:
{{text}}
    """

    def __init__(self, client: GeminiClient):
        self.client = client

    def chunk(self, text: str) -> list[Chunk]:
        lines = text.splitlines()
        number_of_lines = len(lines)
        numbered_lines = [f"{i} {line}" for i, line in enumerate(lines)]

        text = '\n'.join(numbered_lines)

        prompt = self.prompt.format(text=text)

        try:
            response = self.client(prompt)
            lines = response.strip().split('\n')

            # 检查是否每一行都是两个整数
            ranges: list[tuple[int, int]] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: '{line}'")
                start, end = map(int, parts)
                ranges.append((start, end))

            # 检查是否没有重叠或遗漏
            for i in range(1, len(ranges)):
                if ranges[i - 1][1] + 1 != ranges[i][0]:
                    raise ValueError(f"Invalid line format: '{ranges[i - 1]}, {ranges[i]}'")
            if ranges[0][0] != 0 or ranges[-1][1] != number_of_lines - 1:
                raise ValueError("First paragraph must start at line 0 and last must end at final line.")

        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}")

        return [
            Chunk('\n'.join(lines[chunk[0]:chunk[1] + 1]))
            for chunk in ranges
        ]
