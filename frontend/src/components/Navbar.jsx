const Navbar = () => {
  return (
    <nav className="flex items-center justify-between pt-5">
      <img src="src\assets\logo.jpg" className="w-40 ml-0=28"></img>
      <ul>
        <li className="text-gray-600 font-arimo text-xl">
          <a href="#" className="font-arimo m-5 p-2 hover:bg-gradient-to-r from-blue-200 via-purple-300 to-yellow-300 hover:rounded-full ">Home</a>
          <a href="#" className="font-arimo m-5 p-2 hover:bg-gradient-to-r from-blue-200 via-purple-300 to-yellow-300 hover:rounded-full ">Create music</a>
          <a href="#" className="font-arimo m-5 p-2 hover:bg-gradient-to-r from-blue-200 via-purple-300 to-yellow-300 hover:rounded-full " >Log in </a>
          <a href="#" className="font-arimo m-5 p-2 hover:bg-gradient-to-r from-blue-200 via-purple-300 to-yellow-300 hover:rounded-full ">Sign up</a>
        </li>
      </ul>

    </nav>
  );
};

export default Navbar;
