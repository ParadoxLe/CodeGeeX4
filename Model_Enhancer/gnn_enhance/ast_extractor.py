from typing import List
import tree_sitter
import tree_sitter_python as tspython

class ASTExtractor:
    """抽象语法树（AST）提取器"""
    def __init__(self):
        self.lang = tree_sitter.Language(tspython.language(), "python")
        self.parser = tree_sitter.Parser()
        self.parser.set_language(self.lang)

    def _traverse_ast(self, node: tree_sitter.Node, depth: int = 0) -> List[str]:
        """遍历AST提取语法结构"""
        descriptions = []
        indent = "  " * depth
        node_type = node.type.replace("_", " ")
        # 关注关键语法节点
        if node_type in ["function definition", "class definition", "if statement", "for statement"]:
            node_text = node.text.decode().split("\n")[0][:50]
            descriptions.append(f"{indent}- {node_type}：{node_text}")
        # 递归子节点
        for child in node.children:
            descriptions.extend(self._traverse_ast(child, depth + 1))
        return descriptions

    def get_ast_structure(self, code_snippet: str) -> str:
        """生成AST结构提示词"""
        if not code_snippet.strip():
            return ""
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        structure = self._traverse_ast(tree.root_node)
        return "语法结构：\n" + "\n".join(structure) if structure else ""

def get_ast_prompt(code_snippet: str) -> str:
    """对外接口：生成AST增强提示词"""
    extractor = ASTExtractor()
    return extractor.get_ast_structure(code_snippet)