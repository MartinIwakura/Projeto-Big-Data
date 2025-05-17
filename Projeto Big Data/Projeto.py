import pandas as pd
from sklearn import tree, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# VISUALIZAR OS DADOS
def mostrar_dados():
    try:
        df = pd.read_csv('DadosVen.csv')
        print("\n--- Visualização dos Dados ---")
        print(df.head(120))  
        print(f"\nTotal de registros: {len(df)}")
        print(f"Colunas disponíveis: {list(df.columns)}")
    except Exception as e:
        print("Erro ao carregar os dados:", e)


# TESTAR OS DESEMPENHOS (ACCURACY)

# Testar desempenho do valor
def testar_desempenho_valor():
  # leitura dos dados
  df = pd.read_csv('DadosVen.csv')

  # aplicar get_dummies para colunas categóricas
  df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

  # Definir apenas uma coluna de entrada 'Valor' e 'Status' como saída
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
  X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]  # Usando as colunas geradas por get_dummies para 'Parcelas'
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
  df = pd.get_dummies(df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

  # Definir X (entrada) e y (saída)
  X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
  y = df['Status']  # 'Status' é a variável alvo

  # divisão entre treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  # aprendizado da árvore de decisão
  clf = tree.DecisionTreeClassifier()
  clf.fit(X_train, y_train)

  # prever e calcular a acurácia
  y_prediction = clf.predict(X_test)
  print("Acurácia:", accuracy_score(y_test, y_prediction))



# MOSTRAR AS ARVORES DE DECISÃO

# Arvore Valor
def mostrar_arvore_decisao_Valor():
    # Leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Codificar 'Status' se for categórico (opcional, só se for string)
    if df['Status'].dtype == object:
        df['Status'] = df['Status'].map({'Pago': 1, 'Pendente': 0})

    # Apenas a coluna Valor como entrada
    X = df[['Valor']]
    y = df['Status']

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Criar e treinar o classificador
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Plotar a árvore
    plt.figure(figsize=(10, 6))
    tree.plot_tree(clf, filled=True, feature_names=['Valor'], class_names=['Pendente', 'Pago'])
    plt.title("Árvore de Decisão - Valor")
    plt.show()


# Arvore Parcelas
def mostrar_arvore_decisao_Parcelas():
  # leitura dos dados
  df = pd.read_csv('DadosVen.csv')

  # aplicar get_dummies para colunas categóricas, incluindo 'Parcelas'
  df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

  # Definir a coluna 'Parcelas' como entrada e 'Status' como saída
  X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]  # Usando as colunas geradas por get_dummies para 'Parcelas'
  y = df['Status']  # 'Status' é a variável alvo

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  clf = tree.DecisionTreeClassifier()
  clf.fit(X_train, y_train)

  plt.figure(figsize=(5, 5))
  tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
  plt.title("Árvore de Decisão - Parcelas")
  plt.show()



# Arvore Forma de Pagamento
def mostrar_arvore_decisao_forma_pagamento():
  # leitura dos dados
  df = pd.read_csv('DadosVen.csv')

  # Aplicando o One-Hot Encoding para as colunas categóricas
  df = pd.get_dummies(df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

  # Definir X (entrada) e y (saída)
  X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
  y = df['Status']  # 'Status' é a variável alvo

  # divisão entre treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  # aprendizado da árvore de decisão
  clf = tree.DecisionTreeClassifier()
  clf.fit(X_train, y_train)

  # Exibindo a árvore de decisão
  plt.figure(figsize=(5, 5))
  tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
  plt.title("Árvore de Decisão - Forma de Pagamento")
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    classificar_nova_entrada(clf, X.columns)

     # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))



# TESTAR SVM

# Testar Desempenho do Valor com SVM
def testar_svm_valor():
    # Leitura dos dados
    df = pd.read_csv('DadosVen.csv')

    # Se o 'Status' for categórico (strings), converte para numérico
    if df['Status'].dtype == object:
        df['Status'] = df['Status'].map({'Pago': 1, 'Pendente': 0})

    # Usar apenas a coluna 'Valor' como entrada
    X = df[['Valor']]
    y = df['Status']

    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Criar e treinar o modelo SVM
    clf = svm.SVC(gamma = 'auto')
    clf.fit(X_train, y_train)

    # Fazer previsões
    y_pred = clf.predict(X_test)

    # Avaliar acurácia
    print("Acurácia:", accuracy_score(y_test, y_pred))



# Testar o Desempenho das Parcelas com SVM
def testar_desempenho_parcelas_svm():
  # leitura dos dados
  df = pd.read_csv('DadosVen.csv')

  # aplicar get_dummies para colunas categóricas, incluindo 'Parcelas'
  df = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

  # Definir a coluna 'Parcelas' como entrada e 'Status' como saída
  X = df[['Parcelas_A Vista', 'Parcelas_10x', 'Parcelas_12x']]  # Usando as colunas geradas por get_dummies para 'Parcelas'
  y = df['Status']  # 'Status' é a variável alvo

  # divisão entre treino e teste
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=1
  )

  # Treinamento do modelo
  clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
  clf.fit(X_train, y_train)

  # mostrar desempenho
  y_prediction = clf.predict(X_test)
  print("Acurácia:", accuracy_score(y_test, y_prediction))



# Testar o Desempenho das Formas de Pagamento com SVM
def testar_desempenho_forma_pagamento_svm():
  # leitura dos dados
  df = pd.read_csv('DadosVen.csv')

  # Aplicando o One-Hot Encoding para as colunas categóricas
  df = pd.get_dummies(df, columns=['Parcelas', 'Forma_Pagamento'], drop_first=True)

  # Definir X (entrada) e y (saída)
  X = df.drop('Status', axis=1)  # Remover a coluna 'Status' (saída)
  y = df['Status']  # 'Status' é a variável alvo

  # divisão entre treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  # Treinamento do modelo
  clf = make_pipeline(StandardScaler(), svm.SVC(gamma = 'auto'))
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
        forma_pagamento = input("Digite a forma de pagamento (TRANSF. BC., CARTAO, TRANSF. BC. + VALE, CARTAO + VALE, DINHEIRO + CARTAO + VALE, DINHEIRO + VALE, DINHEIRO + CARTAO): ")
        parcelas = input("Digite o número de parcelas (A Vista, 10x, 12x): ")

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


def classificar_arvore_svm():
    df = pd.read_csv('DadosVen.csv')
    df_encoded = pd.get_dummies(df, columns=['Forma_Pagamento', 'Parcelas'])

    df_encoded['Status'] = df_encoded['Status'].map({'Pago': 1, 'Pendente': 0})

    X = df_encoded.drop('Status', axis=1)
    y = df_encoded['Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Treinamento do modelo
    clf = svm.SVC(gamma = 'auto')
    clf.fit(X_train, y_train)

    classificar_nova_entrada_svm(clf, X.columns)

     # mostrar desempenho
    y_prediction = clf.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_prediction))




# MENU INTERATIVO

def mostrar_menu():
    print("1. Visualizar Dados")
    print("2. Arvore de Decisão")
    print("3. SVM")
    print("4. Sair")

def arvore_decisao():
    print("\n** Árvore de Decisão **")
    while True:
        print("\n1. Mostrar os desempenhos")
        print("2. Mostrar as árvores")
        print("3. Classificar novas entradas")
        print("4. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == '1':
          print("\n---------- Desempenho do Valor ----------")
          testar_desempenho_valor()
          print("\n---------- Desempenho das Parcelas ----------")
          testar_desempenho_parcelas()
          print("\n---------- Desempenho das Formas de Pagamento ----------")
          testar_desempenho_forma_pagamento()
          print("\n--------------------------------------------------------")

        elif opcao == '2':
            while True:
                print("\nQual árvore de decisão deseja visualizar?")
                print("1. Árvore de Valor")
                print("2. Árvore de Parcelas")
                print("3. Árvore de Forma de Pagamento")
                print("4. Voltar")

                escolha = input("Escolha uma opção: ")

                if escolha == '1':
                    print("\nExibindo a Árvore de Valor:")
                    mostrar_arvore_decisao_Valor()
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
          print("\n---------- Desempenho do Valor com SVM ----------")
          testar_svm_valor()
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
          mostrar_dados()
        elif opcao == '2':
            arvore_decisao()
        elif opcao == '3':
            executar_svm()
        elif opcao == '4':
            print("Saindo do programa...")
            break
        else:
            print("Opção inválida, tente novamente.")

if __name__ == "__main__":
    main()
