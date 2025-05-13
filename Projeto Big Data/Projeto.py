import pandas as pd
from sklearn import tree, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# TESTAR OS DESEMPENHOS (ACCURACY)

# Testar desempenho do Status


def testar_desempenho_status():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # aplicar get_dummies para colunas categóricas
    df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir apenas uma coluna de entrada (por exemplo, 'Valor') e 'Status' como saída
    X = df[['Valor']]  # Usando apenas a coluna Valor como entrada
    y = df['Status']   # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # aprendizado da árvore de decisão
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# Testar desempenho Parcelas
def testar_desempenho_parcelas():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # aplicar get_dummies para colunas categóricas, incluindo 'Parcelas'
    df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir a coluna 'Parcelas' como entrada e 'Status' como saída
    # Usando as colunas geradas por get_dummies para 'Parcelas'
    X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]
    y = df['Status']  # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # aprendizado da árvore de decisão
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# Testar desempenho forma_pagamento
def testar_desempenho_forma_pagamento():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Aplicando o One-Hot Encoding para as colunas categóricas
    df = pd.get_dummies(
        df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

    # Definir X (entrada) e y (saída)
    X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
    y = df['Status']  # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    # aprendizado da árvore de decisão
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # prever e calcular a acurácia
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# MOSTRAR AS ARVORES DE DECISÃO

# Arvore Status
def mostrar_arvore_decisao_status():
    df = pd.read_csv('DadosVen.csv')
    df = pd.get_dummies(
        df, columns=['Forma_Pagamento', 'Parcelas'], drop_first=True)

    df['Status'] = df['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df.drop('Status', axis=1)
    y = df['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, filled=True, feature_names=X.columns,
                   class_names=['Pendente', 'Pago'])
    plt.title("Árvore de Decisão - Status")
    plt.show()


# Arvore Parcelas
def mostrar_arvore_decisao_Parcelas():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # aplicar get_dummies para colunas categóricas, incluindo 'Parcelas'
    df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir a coluna 'Parcelas' como entrada e 'Status' como saída
    # Usando as colunas geradas por get_dummies para 'Parcelas'
    X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]
    y = df['Status']  # 'Status' é a variável alvo

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    plt.figure(figsize=(5, 5))
    tree.plot_tree(clf, filled=True, feature_names=X.columns,
                   class_names=clf.classes_)
    plt.title("Árvore de Decisão - Parcelas")
    plt.show()


# Arvore Forma de Pagamento
def mostrar_arvore_decisao_forma_pagamento():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Aplicando o One-Hot Encoding para as colunas categóricas
    df = pd.get_dummies(
        df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

    # Definir X (entrada) e y (saída)
    X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
    y = df['Status']  # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    # aprendizado da árvore de decisão
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Exibindo a árvore de decisão
    plt.figure(figsize=(5, 5))
    tree.plot_tree(clf, filled=True, feature_names=X.columns,
                   class_names=clf.classes_)
    plt.title("Árvore de Decisão - Forma de Pagamento")
    plt.show()


# Classificação
def classificar_nova_entrada(clf, X_columns):
    print("\n** Classificar Nova Entrada **")
    try:
        itens = int(input("Digite o número de itens: "))
        valor = float(input("Digite o valor da venda: "))
        forma_pagamento = input(
            "Digite a forma de pagamento (TRANSF. BC., CARTAO, TRANSF. BC. + VALE, CARTAO + VALE, DINHEIRO + CARTAO + VALE, DINHEIRO + VALE, DINHEIRO + CARTAO): ")
        parcelas = input("Digite o número de parcelas (A Vista, 10x, 12x): ")

        nova_entrada = pd.DataFrame({
            'Itens': [itens],
            'Valor': [valor],
            'Forma_Pagamento_CARTAO': [1 if forma_pagamento == 'CARTAO' else 0],
            'Forma_Pagamento_TRANSF. BC.': [1 if forma_pagamento == 'TRANSF. BC.' else 0],
            'Forma_Pagamento_DINHEIRO': [1 if forma_pagamento == 'DINHEIRO' else 0],
            'Parcelas_A Vista': [1 if parcelas == 'A Vista' else 0],
            'Parcelas_10x': [1 if parcelas == '10x' else 0],
            'Parcelas_12x': [1 if parcelas == '12x' else 0]
        })

        nova_entrada = nova_entrada.reindex(columns=X_columns, fill_value=0)

        predicao = clf.predict(nova_entrada)
        print(f"Classificação: {'Pago' if predicao[0] == 1 else 'Pendente'}")

    except Exception as e:
        print("Erro ao inserir os dados:", e)


def classificar_arvore():
    df = pd.read_csv('DadosVen.csv')
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    df_encoded['Status'] = df_encoded['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    classificar_nova_entrada(clf, X.columns)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# TESTAR SVM

# Testar Desempenho do Status com SVM
def testar_desempenho_status_svm():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # aplicar get_dummies para colunas categóricas
    df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir apenas uma coluna de entrada (por exemplo, 'Valor') e 'Status' como saída
    X = df[['Valor']]  # Usando apenas a coluna Valor como entrada
    y = df['Status']   # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # aprendizado do SVC
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# Testar o Desempenho das Parcelas com SVM
def testar_desempenho_parcelas_svm():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # aplicar get_dummies para colunas categóricas, incluindo 'Parcelas'
    df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir a coluna 'Parcelas' como entrada e 'Status' como saída
    # Usando as colunas geradas por get_dummies para 'Parcelas'
    X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]
    y = df['Status']  # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Treinamento do modelo
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# Testar o Desempenho das Formas de Pagamento com SVM
def testar_desempenho_forma_pagamento_svm():
    # leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Aplicando o One-Hot Encoding para as colunas categóricas
    df = pd.get_dummies(
        df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

    # Definir X (entrada) e y (saída)
    X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
    y = df['Status']  # 'Status' é a variável alvo

    # divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    # Treinamento do modelo
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)

    # prever e calcular a acurácia
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# Classificar novas entradas com SVM
def classificar_nova_entrada_svm(clf, X_columns):
    print("\n** Classificar Nova Entrada **")
    try:
        itens = int(input("Digite o número de itens: "))
        valor = float(input("Digite o valor da venda: "))
        forma_pagamento = input(
            "Digite a forma de pagamento (TRANSF. BC., CARTAO, TRANSF. BC. + VALE, CARTAO + VALE, DINHEIRO + CARTAO + VALE, DINHEIRO + VALE, DINHEIRO + CARTAO): ")
        parcelas = input("Digite o número de parcelas (A Vista, 10x, 12x): ")

        nova_entrada = pd.DataFrame({
            'Itens': [itens],
            'Valor': [valor],
            'Forma_Pagamento_CARTAO': [1 if forma_pagamento == 'CARTAO' else 0],
            'Forma_Pagamento_TRANSF. BC.': [1 if forma_pagamento == 'TRANSF. BC.' else 0],
            'Forma_Pagamento_DINHEIRO': [1 if forma_pagamento == 'DINHEIRO' else 0],
            'Parcelas_A Vista': [1 if parcelas == 'A Vista' else 0],
            'Parcelas_10x': [1 if parcelas == '10x' else 0],
            'Parcelas_12x': [1 if parcelas == '12x' else 0]
        })

        nova_entrada = nova_entrada.reindex(columns=X_columns, fill_value=0)

        predicao = clf.predict(nova_entrada)
        print(f"Classificação: {'Pago' if predicao[0] == 1 else 'Pendente'}")

    except Exception as e:
        print("Erro ao inserir os dados:", e)


def classificar_arvore_svm():
    df = pd.read_csv('DadosVen.csv')
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    df_encoded['Status'] = df_encoded['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    # Treinamento do modelo
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)

    classificar_nova_entrada(clf, X.columns)

    # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))


# MENU INTERATIVO

def mostrar_menu():
    print("1. Arvore de Decisão")
    print("2. SVM")
    print("3. Sair")


def arvore_decisao():
    print("\n** Árvore de Decisão **")
    while True:
        print("\n1. Mostrar os desempenhos")
        print("2. Mostrar as árvores")
        print("3. Classificar novas entradas")
        print("4. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            print("\n---------- Desempenho do Status ----------")
            testar_desempenho_status()
            print("\n---------- Desempenho das Parcelas ----------")
            testar_desempenho_parcelas()
            print("\n---------- Desempenho das Formas de Pagamento ----------")
            testar_desempenho_forma_pagamento()
            print("\n--------------------------------------------------------")

        elif opcao == '2':
            while True:
                print("\nQual árvore de decisão deseja visualizar?")
                print("1. Árvore de Status")
                print("2. Árvore de Parcelas")
                print("3. Árvore de Forma de Pagamento")
                print("4. Voltar")

                escolha = input("Escolha uma opção: ")

                if escolha == '1':
                    print("\nExibindo a Árvore de Status:")
                    mostrar_arvore_decisao_status()
                elif escolha == '2':
                    print("\nExibindo a Árvore de Parcelas:")
                    mostrar_arvore_decisao_Parcelas()
                elif escolha == '3':
                    print("\nExibindo a Árvore de Forma de Pagamento:")
                    mostrar_arvore_decisao_forma_pagamento()
                elif escolha == '4':
                    break
                else:
                    print("Opção inválida, tente novamente.")

        elif opcao == '3':
            print("Classificando uma nova entrada:")
            print("")
            classificar_arvore()

        elif opcao == '4':
            print("Saindo da Árvore de Decisão...")
            break
        else:
            print("Opção inválida, tente novamente.")


def executar_svm():
    print("\n** SVM **")
    while True:
        print("\n1. Mostrar os desempenhos")
        print("2. Classificar nova entrada")
        print("3. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            print("\n---------- Desempenho do Status com SVM ----------")
            testar_desempenho_status_svm()
            print("\n---------- Desempenho das Parcelas com SVM ----------")
            testar_desempenho_parcelas_svm()
            print("\n---------- Desempenho das Formas de Pagamento com SVM ----------")
            testar_desempenho_forma_pagamento_svm()
            print("\n--------------------------------------------------------")

        elif opcao == '2':
            print("Classificando uma nova entrada com SVM:")
            classificar_arvore_svm()

        elif opcao == '3':
            print("Saindo do SVM...")
            break
        else:
            print("Opção inválida, tente novamente.")


def main():
    while True:
        mostrar_menu()
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            arvore_decisao()
        elif opcao == '2':
            executar_svm()
        elif opcao == '3':
            print("Saindo do programa...")
            break
        else:
            print("Opção inválida, tente novamente.")


if __name__ == "__main__":
    main()
