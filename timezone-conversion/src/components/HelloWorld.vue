<template>
  <div class="container">
    <h1>Conversão de Fusos Horários</h1>
    <label for="user-date">Data e Hora (Local):</label>
    <input type="datetime-local" v-model="userDate" />
    <br />
    <button @click="convertAndSubmit">Enviar</button>

    <div v-if="convertedDate">
      <h3>Data Convertida para São Paulo:</h3>
      <p>{{ convertedDate }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      userDate: "", // Armazena a data inserida pelo usuário
      convertedDate: null, // Armazena a data convertida para o fuso de São Paulo
    };
  },
  methods: {
    convertAndSubmit() {
      if (!this.userDate) {
        alert("Por favor, insira uma data e hora.");
        return;
      }

      // Converte a entrada para um objeto Date
      const userDate = new Date(this.userDate);
      console.log("Data inserida pelo usuário:", this.userDate);

      // Detecta automaticamente o fuso horário do usuário
      const userTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      console.log("Fuso horário do usuário:", userTimeZone);

      // Fuso horário de São Paulo
      const saoPauloTimeZone = "America/Sao_Paulo";

      // Converte o horário do usuário para UTC usando seu fuso horário local
      const userTzDate = new Date(
        userDate.toLocaleString("en-US", { timeZone: userTimeZone })
      );

      // Converte o horário UTC para o fuso horário de São Paulo
      const saoPauloTzDate = new Date(
        userTzDate.toLocaleString("en-US", { timeZone: saoPauloTimeZone })
      );

      // Converte para o formato ISO (UTC) para enviar ao backend
      this.convertedDate = saoPauloTzDate;

      // Exibe a data convertida
      console.log(
        "Data convertida para o fuso de São Paulo:",
        this.convertedDate
      );
    },
  },
};
</script>

<style scoped>
.container {
  padding: 20px;
}
button {
  margin-top: 10px;
}
</style>
