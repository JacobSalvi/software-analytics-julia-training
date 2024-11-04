from tree_sitter_languages import get_language, get_parser


def parse():
    language = get_language('python')
    parser = get_parser('python')
    example = """
a = 1

'''This
is
not
a
multiline
comment.'''

b = 2

class Test:
    "This is a class docstring."

    'This is bogus.'

    def test(self):
        "This is a function docstring."

        "Please, no."

        return 1

c = 3
    """
    tree = parser.parse(example.encode())
    node = tree.root_node
    print(node.sexp())
    pass


if __name__ == '__main__':
    parse()