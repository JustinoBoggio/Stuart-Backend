// recovery.js

// Variables simulando la existencia de usuarios
const registeredEmails = ["usuario@ejemplo.com", "otro@ejemplo.com"];

// Almacena el código generado y el temporizador
let generatedCode = "";
let countdown;
let timeLeft = 300; // 5 minutos en segundos
let email;

// Maneja el submit del formulario
document.getElementById("recovery-form").addEventListener("submit", async function(event) {
  event.preventDefault();

  // 1) Mostramos el modal de carga antes de comenzar el fetch
  document.getElementById("loading-modal").style.display = "flex";

  email = document.getElementById("recovery-email").value;

  // 1) Verificamos la existencia del correo
  try {
    const responseCheck = await fetch('http://localhost:5000/api/checkEmail', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });

    const resultCheck = await responseCheck.json();

    if (!responseCheck.ok) {
      // No existe o hay algún error
      console.error(resultCheck.error);
      //Ocultamos el modal de carga
      document.getElementById("loading-modal").style.display = "none";
      // Muestra el modal de advertencia
      document.getElementById("warning-modal").style.display = "flex";
      return;
    }
    else{
      console.log(resultCheck.message); // 'El correo existe'

      // 2) Generamos el código en el FRONT
      const code = generateCode(6);
      generatedCode = code;

      // 3) Llamamos al endpoint para enviar el correo
      const responseMail = await fetch('http://localhost:5000/api/sendMail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code }),
      });

      const resultMail = await responseMail.json();

      if (responseMail.ok) {
        // Éxito al enviar
        console.log(resultMail.message);
        // 5) Ahora que tenemos la respuesta de sendMail, ocultamos el modal de carga
        document.getElementById("loading-modal").style.display = "none";
        //alert(`Código enviado al correo: ${email}`);
        document.getElementById("success-modal-SendMail").style.display = "flex";
        // Aquí puedes abrir tu pop-up o redirigir
        // Muestra la ventana popUp de código
        openCodePopup();
      } else {
        // Error al enviar
        console.error(resultMail.error);
        // 5) Ahora que tenemos la respuesta de sendMail, ocultamos el modal de carga
        document.getElementById("loading-modal").style.display = "none";
        document.getElementById("error-modal-FailSend").style.display = "flex";
        //alert(resultMail.error);
      }
    }
  }
  catch (error) {
  console.error("Error de conexión:", error);
  //alert("Error de conexión con el servidor");
  }
});

// Cierra modal genérico
function closeModal(modalId) {
  document.getElementById(modalId).style.display = "none";
}

// Genera un código aleatorio de 'length' dígitos
function generateCode(length) {
  let code = "";
  for (let i = 0; i < length; i++) {
    code += Math.floor(Math.random() * 10); // dígito aleatorio [0..9]
  }
  return code;
}

// Abre el popUp de código y arranca el temporizador
function openCodePopup() {
  document.getElementById("code-popup").style.display = "flex";
  startCountdown();
}

// Inicia el conteo regresivo
function startCountdown() {
  const countdownTimer = document.getElementById("countdown-timer");

  countdown = setInterval(() => {
    if (timeLeft <= 0) {
      clearInterval(countdown);
      // Cierra popUp y obliga a reenviar código
      //document.getElementById("code-popup").style.display = "none";
      //alert("Se acabó el tiempo, por favor solicita un nuevo código.");
      document.getElementById("error-time-ended-modal").style.display = "flex";
    } else {
      // Calcula minutos y segundos
      const minutes = Math.floor(timeLeft / 60);
      const seconds = timeLeft % 60;
      countdownTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
      timeLeft--;
    }
  }, 1000);
}

// Verifica el código ingresado
function verifyCode() {
  // Toma los 6 inputs
  const digits = document.querySelectorAll(".code-digit");
  let enteredCode = "";
  digits.forEach((digit) => (enteredCode += digit.value));

  if (enteredCode === generatedCode) {
    // Limpia el temporizador y cierra popUp
    clearInterval(countdown);
    document.getElementById("code-popup").style.display = "none";

    // Muestra popUp de nueva contraseña
    document.getElementById("new-pass-popup").style.display = "flex";
  } else {
    // Muestra modal de error
    document.getElementById("error-code-modal").style.display = "flex";
  }
}

document.getElementById('new-pass-form').addEventListener('submit', async function(event) {
  event.preventDefault();
  const newPass = document.getElementById("newPass").value;
  const confirmNewPass = document.getElementById("confirmNewPass").value;

  if (newPass === confirmNewPass && newPass !== "") {
    
    try
    {
      const response = await fetch('http://localhost:5000/api/UpdatePassword', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: email, password: newPass })
      });
  
      const result = await response.json();
  
      if (response.ok){
        document.getElementById('success-modal').style.display = 'flex';
      }
      else{
        document.getElementById("error-modal-ErroSavePassword").style.display = "flex";
      }
    }
    catch(error)
    {
      document.getElementById("error-modal-ErroSavePassword").style.display = "flex";
    }
  } else {
    // Muestra modal de error para contraseñas
    document.getElementById("error-modal").style.display = "flex";
  }
});

// Cierra modal de éxito
function closeModalSuccess(modalId) {
    document.getElementById(modalId).style.display = "none";

    // Cierra el popUp y redirige al login
    document.getElementById("new-pass-popup").style.display = "none";

    // Redirige a la página de LogIn
    window.location.href = "index.html";
}

// Cierra modal de Tiempo Finalizado
function closeModalTimenEnded(modalId) {
    document.getElementById(modalId).style.display = "none";

    // Cierra el popUp de Ingreso de Código
    document.getElementById("code-popup").style.display = "none";
}
  