nota = int(input("Por favor adicione aqui uma nota de 0 a 100: "))

if 0 <= nota <= 100:
  if nota <= 69:
   print("Atendimento \033[1;31minsatisfatório\033[0m!")
  elif nota >= 70 and nota <= 89:
   print("Atendimento \033[1;36mneutro\033[0m!")
  elif nota >= 90 and nota <= 100:
   print("Atendimento de \033[0;32mqualidade\033[0m!")
else:
  print("Nota inválida!")