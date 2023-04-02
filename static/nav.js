var checkbox = document.getElementById("my-checkbox");
  checkbox.addEventListener('change', function() {
    if (this.checked) {
      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/process_checkbox', true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.send(JSON.stringify({'value': true}));
    }
  });

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiPGFub255bW91cz4iXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7O0FBQUEiLCJzb3VyY2VzQ29udGVudCI6WyIiXX0=
//# sourceURL=coffeescript