from typing import List, Dict, Tuple
import tree_sitter
import tree_sitter_python as tspython


class CFGExtractor:
    """控制流图（CFG）提取器"""

    def __init__(self):
        self.lang = tree_sitter.Language(tspython.language(), "python")
        self.parser = tree_sitter.Parser()
        self.parser.set_language(self.lang)
        self.nodes = []  # 控制流节点
        self.edges = []  # 控制流边
        self.node_id = 0

    def _add_node(self, node_type: str, content: str) -> int:
        """添加控制流节点（去重）"""
        self.nodes.append({"id": self.node_id, "type": node_type, "content": content})
        self.node_id += 1
        return self.node_id - 1

    def _traverse_ast(self, node: tree_sitter.Node, parent_id: int):
        """遍历AST提取控制流关系"""
        if node.type in ["if_statement", "for_statement", "while_statement"]:
            # 控制流语句节点
            stmt_node_id = self._add_node(node.type, node.text.decode())
            self.edges.append((parent_id, stmt_node_id))

            # 处理分支/循环体
            if node.type == "if_statement":
                then_branch = node.child_by_field_name("consequence")
                else_branch = node.child_by_field_name("alternative")
                if then_branch:
                    then_id = self._add_node("then_block", then_branch.text.decode())
                    self.edges.append((stmt_node_id, then_id))
                    self._traverse_ast(then_branch, then_id)
                if else_branch:
                    else_id = self._add_node("else_block", else_branch.text.decode())
                    self.edges.append((stmt_node_id, else_id))
                    self._traverse_ast(else_branch, else_id)
            return stmt_node_id

        # 普通语句节点
        if node.type not in ["module", "block", "expression_statement"]:
            stmt_id = self._add_node("statement", node.text.decode())
            self.edges.append((parent_id, stmt_id))
            parent_id = stmt_id

        # 递归处理子节点
        for child in node.children:
            self._traverse_ast(child, parent_id)

    def extract(self, code_snippet: str) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """提取CFG节点和边"""
        self.nodes = []
        self.edges = []
        self.node_id = 0
        if not code_snippet.strip():
            return [], []
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        root_id = self._add_node("root", "程序入口")
        self._traverse_ast(tree.root_node, root_id)
        return self.nodes, self.edges


def get_control_flow_prompt(code_snippet: str) -> str:
    """生成控制流提示词"""
    extractor = CFGExtractor()
    nodes, edges = extractor.extract(code_snippet)
    if not nodes or not edges:
        return ""

    node_map = {n["id"]: n for n in nodes}
    flow_desc = []
    for src_id, tgt_id in edges:
        src = node_map[src_id]
        tgt = node_map[tgt_id]
        if src["type"] == "root":
            flow_desc.append(f"程序开始执行：{tgt['content'][:50]}")
        elif src["type"] == "if_statement":
            flow_desc.append(f"条件判断「{src['content'][:30]}」后执行：{tgt['content'][:50]}")
        elif src["type"] == "for_statement":
            flow_desc.append(f"循环「{src['content'][:30]}」中执行：{tgt['content'][:50]}")

    return "控制流逻辑：" + "；".join(flow_desc) + "。" if flow_desc else ""