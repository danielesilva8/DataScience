import sqlite3


class Gestao: 
    def __init__(self, banco):
        self.conn = sqlite3.connect(banco)
        self.criar_tabela_estoque()

    def criar_tabela_estoque(self):
        cursor = self.conn.cursor()
        cursor.execute('''
              CREATE TABLE IF NOT EXISTS estoque (
            id INTEGER PRIMARY KEY,
            produto TEXT,
            quantidade INTEGER
              )
        ''')
        self.conn.commit()

    def adicionar_produto(self, produto, quantidade):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO estoque (produto, quantidade) VALUES (?, ?)", (produto, quantidade))
        self.conn.commit()
 
    def remover_produto(self, produto, quantidade):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT quantidade FROM estoque WHERE produto=?", (produto,))
        resultado = cursor.fetchone()
        if resultado:
            estoque_atual = resultado[0]
            if estoque_atual >= quantidade:
                cursor.execute("UPDATE estoque SET quantidade=? WHERE produto=?",
                              (estoque_atual - quantidade, produto))
                self.conn.commit()
            else:
                print(f"Quantidade insuficiente de {produto} em estoque.")
        else:
            print(f"{produto} não encontrado em estoque.") 
    def consultar_estoque(self, produto): 
        cursor = self.conn.cursor()
        cursor.execute(
              "SELECT quantidade FROM estoque WHERE produto=?", (produto,))
        resultado = cursor.fetchone()
        if resultado:
            return resultado[0]
        else: 
            return 0

    def listar_produtos(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT produto FROM estoque")
        produtos = cursor.fetchall()
        return [produto[0] for produto in produtos]     



sistema = Gestao("estoque.db")

sistema.adicionar_produto("Camiseta", 50)
sistema.adicionar_produto("Calça Jeans", 30)
sistema.adicionar_produto("Tênis", 20)        


estoque_camiseta = sistema.consultar_estoque("Camiseta")
print(f"Quantidade de Camisetas em estoque: {estoque_camiseta}")


sistema.remover_produto("Calça Jeans", 10)
print('Estoque de calças alterado com sucesso')

produtos_em_estoque = sistema.listar_produtos()
print("Produtos em estoque:", produtos_em_estoque)
