import pandas as pd
from sklearn import tree, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Testar Desempenho
def testar_desempenho():
    # Leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Converter variáveis categóricas para numéricas (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir variáveis de entrada (X) e saída (y)
    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    # Mapear 'Status' para valores numéricos
    y = y.map({'Pago': 1, 'Pendente': 0})

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Treinar modelo
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Fazer previsões e medir desempenho
    y_prediction = clf.predict(X_test)
    acuracia = accuracy_score(y_test, y_prediction) * 100
    print("\nDesempenho da Árvore de Decisão:")
    print("Acurácia: {:.2f}%".format(acuracia))


# Mostrar Arvore
def mostrar_arvore():
    # Leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Converter variáveis categóricas para numéricas (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir variáveis de entrada (X) e saída (y)
    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    # Mapear 'Status' para valores numéricos
    y = y.map({'Pago': 1, 'Pendente': 0})

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Treinar modelo
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Plotar a árvore com os nomes corretos das features
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['Pendente', 'Pago'])
    plt.title("Árvore de Decisão")
    plt.show()


# Classificação
def classificar_nova_entrada(clf, X_columns):
    print("\n** Classificar Nova Entrada **")
    try:
        itens = int(input("Digite o número de itens: "))
        valor = float(input("Digite o valor da venda: "))
        forma_pagamento = input("Digite a forma de pagamento (TRANSF. BC., CARTAO, TRANSF. BC. + VALE, CARTAO + VALE, DINHEIRO + CARTAO + VALE, DINHEIRO + VALE, DINHEIRO + CARTAO): ")
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

        # Garante que todas as colunas esperadas estão presentes e ordenadas corretamente
        nova_entrada = nova_entrada.reindex(columns=X_columns, fill_value=0)

        predicao = clf.predict(nova_entrada)
        print(f"Classificação: {'Pago' if predicao[0] == 1 else 'Pendente'}")

    except Exception as e:
        print("Erro na classificação:", e)


def executar_classificacao():
    df = pd.read_csv('DadosVen.csv')
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    df_encoded['Status'] = df_encoded['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    classificar_nova_entrada(clf, X.columns)


# TESTAR SVM

# Testar Desempenho com SVM
def testar_svm():
    # Leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Converter variáveis categóricas para numéricas (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    # Definir variáveis de entrada (X) e saída (y)
    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    # Mapear 'Status' para valores numéricos
    y = y.map({'Pago': 1, 'Pendente': 0})

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Treinamento do modelo com SVM
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)


    # Fazer previsões e medir desempenho
    y_prediction = clf.predict(X_test)
    acuracia = accuracy_score(y_test, y_prediction) * 100
    print("\nDesempenho do SVM:")
    print("Acurácia: {:.2f}%".format(acuracia))




# Classificar novas entradas com SVM
def classificar_nova_entrada_svm(clf, X_columns):
    print("\n** Classificar Nova Entrada **")
    try:
        itens = int(input("Digite o número de itens: "))
        valor = float(input("Digite o valor da venda: "))
        forma_pagamento = input("Digite a forma de pagamento (TRANSF. BC., CARTAO,TRANSF. BC. + VALE, CARTAO + VALE, DINHEIRO + CARTAO + VALE, DINHEIRO + VALE, DINHEIRO + CARTAO): ").upper()

        parcelas = input("Digite o número de parcelas (A Vista, 10x, 12x): ").upper()

        # Criar DataFrame com zeros para todas as colunas
        nova_entrada = pd.DataFrame(columns=X_columns)
        nova_entrada.loc[0] = 0  # preencher com zeros

        # Preencher colunas conhecidas
        if 'Itens' in X_columns:
            nova_entrada.at[0, 'Itens'] = itens
        if 'Valor' in X_columns:
            nova_entrada.at[0, 'Valor'] = valor

        # Mapear forma_pagamento para as colunas dummy
        forma_map = {
            'CARTAO': 'Forma_Pagamento_CARTAO',
            'TRANSF. BC.': 'Forma_Pagamento_TRANSF. BC.',
            'DINHEIRO': 'Forma_Pagamento_DINHEIRO',
            # Acrescente outras formas conforme dataset
        }
        col_fp = forma_map.get(forma_pagamento.upper(), None)
        if col_fp and col_fp in X_columns:
            nova_entrada.at[0, col_fp] = 1

        # Mapear parcelas
        parcelas_map = {
            'A VISTA': 'Parcelas_A Vista',
            '10X': 'Parcelas_10x',
            '12X': 'Parcelas_12x',
        }
        col_parcelas = parcelas_map.get(parcelas.upper(), None)
        if col_parcelas and col_parcelas in X_columns:
            nova_entrada.at[0, col_parcelas] = 1

        predicao = clf.predict(nova_entrada)
        print(f"Classificação: {'Pago' if predicao[0] == 1 else 'Pendente'}")

    except Exception as e:
        print("Erro ao inserir os dados:", e)


def executar_classificacao_svm():
    df = pd.read_csv('DadosVen.csv')
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    df_encoded['Status'] = df_encoded['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Treinamento do modelo
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    classificar_nova_entrada_svm(clf, X.columns)




# MENU INTERATIVO

def mostrar_menu():
    print("\n---------- Sistema de Classificação de Vendas ----------")
    print("1. Arvore de Decisão")
    print("2. SVM")
    print("3. Sair")

def arvore_decisao():
    print("\n ---------- Árvore de Decisão ----------")
    while True:
        print("\n1. Mostrar o desempenho")
        print("2. Mostrar a árvore")
        print("3. Classificar nova entrada")
        print("4. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
          print("\n---------- Desempenho ----------")
          testar_desempenho()


        elif opcao == '2':
          print("\n ---------- Exibindo a Árvore de Decisão ----------:")
          mostrar_arvore()

        elif opcao == '3':
            print(" ---------- Classificando uma nova entrada:  ----------")
            print("")
            executar_classificacao()

        elif opcao == '4':
            print(" ---------- Saindo da Árvore de Decisão... ----------")
            break
        else:
            print("Opção inválida, tente novamente.")

def executar_svm():
    print("\n--------- SVM ---------")
    while True:
        print("\n1. Mostrar o desempenho")
        print("2. Classificar nova entrada")
        print("3. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            print("\n---------- Desempenho com SVM ----------")
            testar_svm()

        elif opcao == '2':
            print("Classificando uma nova entrada com SVM:")
            executar_classificacao_svm()

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
